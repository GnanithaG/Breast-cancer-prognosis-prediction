#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import set_all_seeds
from src.data.ingest import load_csv
from src.data.harmonize import harmonize_registry_codes
from src.data.splits import make_or_load_splits
from src.preprocessing.pipeline import build_preprocessor
from src.models.cox import train_cox_pipeline
from src.models.rsf import train_rsf_pipeline
from src.models.deepsurv_optional import try_train_deepsurv
from src.evaluation.metrics import evaluate_models
from src.evaluation.plots import (
    plot_calibration_horizons,   # (model_name, model, X, y_surv, horizons, outdir, annotate)
    plot_ipcw_reliability,       # (model_name, model, X, y_surv, horizons, outdir, annotate)
    plot_km_by_risk,             # (model_name, model, X, y_surv, n_groups, outdir, annotate)
)
from src.reporting.save_artifacts import save_metrics, save_model_card


def parse_args():
    p = argparse.ArgumentParser(description="Breast cancer survival pipeline")
    p.add_argument("--data", required=True, help="Path to CSV data")
    p.add_argument("--time_col", default="survival_months")
    p.add_argument("--event_col", default="vital_status")
    p.add_argument("--stage_col", default="stage")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--models", default="cox,rsf", help="comma-separated: cox,rsf,deepsurv")
    p.add_argument("--outdir", default="reports")
    p.add_argument("--cfgdir", default="configs")
    p.add_argument("--horizons", default="12,36,60,120", help="months, comma-separated (e.g., 12,36,60)")
    p.add_argument("--pool_plots", action="store_true", help="Pool VAL+TEST for figures (metrics remain TEST-only)")
    p.add_argument("--annotate", action="store_true", help="Add reader hints/annotations onto figures")
    return p.parse_args()


def to_surv_struct(df: pd.DataFrame, time_col: str, event_col: str):
    from sksurv.util import Surv
    return Surv.from_arrays(
        event=df[event_col].astype(bool).to_numpy(),
        time=df[time_col].astype(float).to_numpy(),
    )


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    cfgdir = Path(args.cfgdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    cfgdir.mkdir(parents=True, exist_ok=True)

    # 1) LOAD
    df = load_csv(args.data)

    # 2) HARMONIZE clinical codes
    df = harmonize_registry_codes(df, stage_col=args.stage_col)

    # 3) SPLITS (70/15/15, stratified)
    split_path = cfgdir / "split_indices.json"
    idx = make_or_load_splits(
        df=df,
        time_col=args.time_col,
        event_col=args.event_col,
        stage_col=args.stage_col,
        save_path=split_path,
        seed=args.seed,
    )
    trn_idx, val_idx, tst_idx = idx["train"], idx["val"], idx["test"]

    # 4) TARGETS/FEATURES
    X = df.drop(columns=[args.time_col, args.event_col])
    y = df[[args.time_col, args.event_col]]

    X_trn, X_val, X_tst = X.iloc[trn_idx], X.iloc[val_idx], X.iloc[tst_idx]
    y_trn_df, y_val_df, y_tst_df = y.iloc[trn_idx], y.iloc[val_idx], y.iloc[tst_idx]

    y_trn_surv = to_surv_struct(y_trn_df, args.time_col, args.event_col)
    y_val_surv = to_surv_struct(y_val_df, args.time_col, args.event_col)
    y_tst_surv = to_surv_struct(y_tst_df, args.time_col, args.event_col)

    # 5) PREPROCESSOR (fit on train only)
    preprocessor, num_cols, cat_cols = build_preprocessor(
        df, time_col=args.time_col, event_col=args.event_col
    )

    # 6) TRAIN MODELS (POSitional args to match your existing functions)
    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not models_to_run:
        raise RuntimeError("No models specified. Use --models cox,rsf[,deepsurv]")

    trained = {}

    if "cox" in models_to_run:
        cox = train_cox_pipeline(
            preprocessor,
            X_trn,
            y_trn_surv,
            X_val,
            y_val_surv,
            random_state=args.seed,
        )
        trained["cox"] = cox

    if "rsf" in models_to_run:
        rsf = train_rsf_pipeline(
            preprocessor,
            X_trn,
            y_trn_surv,
            X_val,
            y_val_surv,
            random_state=args.seed,
        )
        trained["rsf"] = rsf

    if "deepsurv" in models_to_run:
        ds = try_train_deepsurv(
            preprocessor,
            X_trn,
            y_trn_surv,
            X_val,
            y_val_surv,
            random_state=args.seed,
        )
        if ds is not None:
            trained["deepsurv"] = ds

    if not trained:
        raise RuntimeError("No models were trained. Check --models argument.")

    # 7) EVALUATE (TEST-only metrics)
    horizons = np.array([int(h) for h in args.horizons.split(",") if h.strip()], dtype=int)
    metrics = evaluate_models(
        models=trained,
        X_train=X_trn,
        y_train=y_trn_surv,
        X_val=X_val,
        y_val=y_val_surv,
        X_test=X_tst,
        y_test=y_tst_surv,
        horizons=horizons,
    )

    # 8) PLOTS (prefer RSF if available else first)
    plot_model_key = "rsf" if "rsf" in trained else next(iter(trained.keys()))
    model_name = plot_model_key.upper()

    # pool VAL+TEST for clearer figures if requested (metrics remain TEST-only)
    if args.pool_plots:
        print("[plots] Using pooled VAL+TEST for figures only (metrics remain TEST-only).")
        X_plot = pd.concat([X_val, X_tst], axis=0)
        y_plot_df = pd.concat([y_val_df, y_tst_df], axis=0)
        y_plot_surv = to_surv_struct(y_plot_df, args.time_col, args.event_col)
    else:
        X_plot = X_tst
        y_plot_surv = y_tst_surv

    # ---- positional calls to match your plotting function signatures ----
    plot_calibration_horizons(
        model_name,
        trained[plot_model_key],
        X_plot,
        y_plot_surv,
        horizons,
        str(figdir),
        args.annotate,
    )

    plot_ipcw_reliability(
        model_name,
        trained[plot_model_key],
        X_plot,
        y_plot_surv,
        horizons,
        str(figdir),
        args.annotate,
    )

    plot_km_by_risk(
        model_name,
        trained[plot_model_key],
        X_plot,
        y_plot_surv,
        4,
        str(figdir),
        args.annotate,
    )

    # 9) SAVE METRICS + MODEL CARD
    save_metrics(metrics, outdir / "metrics.json")
    save_model_card(
        outdir / "model_card.md",
        models=list(trained.keys()),
        inputs=list(X.columns),
        horizons=horizons.tolist(),
        time_col=args.time_col,
        event_col=args.event_col,
    )

    print(f"Done. See '{outdir}' for metrics and figures.")


if __name__ == "__main__":
    main()
