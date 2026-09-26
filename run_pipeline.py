#!/usr/bin/env python
"""End-to-end breast cancer survival pipeline.

python run_pipeline.py                      # defaults: all models, data/Breast_Cancer.csv
python run_pipeline.py --models cox,rsf --n-boot 0   # quick run, no bootstrap CIs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sksurv.util import Surv

from src.data.harmonize import harmonize_registry_codes
from src.data.ingest import load_csv
from src.data.splits import make_or_load_splits
from src.evaluation.explain import hazard_ratios, is_linear, linear_shap
from src.evaluation.importance import permutation_importance_cindex
from src.evaluation.metrics import evaluate_models, paired_c_index_difference, valid_horizons
from src.evaluation.plots import plot_patient_waterfall, plot_shap_summary, save_all
from src.models.registry import MODELS, fit_and_tune
from src.reporting.save_artifacts import results_table, save_metrics, save_model_card
from src.utils import set_all_seeds

log = logging.getLogger("pipeline")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/Breast_Cancer.csv", help="input CSV")
    p.add_argument("--models", default="stage,cox,rsf,gbsa", help=f"comma-separated, from: {','.join(MODELS)}")
    p.add_argument("--horizons", default="12,36,60,96", help="evaluation horizons in months")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for CIs (0 = off)")
    p.add_argument("--outdir", default="reports")
    p.add_argument("--splits", default="configs/split_indices.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def to_surv(df: pd.DataFrame):
    return Surv.from_arrays(event=df["event"].astype(bool).to_numpy(), time=df["time"].astype(float).to_numpy())


def explain_best(model, label, X, outdir: Path) -> None:
    """SHAP summary, one worked patient example, and a hazard-ratio table."""
    shap_test = linear_shap(model, X["test"], X["train"])
    plot_shap_summary(shap_test, X["test"], label, outdir / "figures" / "shap_summary.png")

    # Worked example: the test patient at the 90th percentile of predicted risk.
    total = shap_test.sum(axis=1)
    pid = (total - total.quantile(0.9)).abs().idxmin()
    plot_patient_waterfall(
        shap_test.loc[pid],
        X["test"].loc[pid],
        f"Why this patient is high-risk (test patient #{pid})",
        outdir / "figures" / "shap_patient_example.png",
    )

    hr = hazard_ratios(model, X["train"])
    (outdir / "hazard_ratios.md").write_text(
        f"# Hazard ratios - {label}\n\n"
        "Categorical levels vs the most common level (reference). Numeric variables per 1 SD. "
        "Ridge-penalised, so estimates are shrunk slightly towards 1.\n\n"
        + hr.to_markdown(index=False, floatfmt=".2f")
        + "\n"
    )


def main(argv=None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    set_all_seeds(args.seed)

    names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = set(names) - set(MODELS)
    if unknown or not names:
        raise SystemExit(f"Unknown/empty --models {sorted(unknown)}; choose from {list(MODELS)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean
    df = harmonize_registry_codes(load_csv(args.data))
    log.info("Loaded %d patients, %d deaths (%.1f%%)", len(df), df["event"].sum(), 100 * df["event"].mean())

    # 2. Frozen split
    split = make_or_load_splits(df, Path(args.splits), seed=args.seed)
    parts = {k: df.loc[split[k]] for k in ("train", "val", "test")}
    X = {k: v.drop(columns=["time", "event"]) for k, v in parts.items()}
    y = {k: to_surv(v) for k, v in parts.items()}
    log.info("Split sizes: %s", split["sizes"])

    horizons = valid_horizons([float(h) for h in args.horizons.split(",")], y["train"], y["test"])
    tau = float(horizons.max())

    # 3. Fit + tune on validation
    fitted, tuning = {}, {}
    for name in names:
        log.info("Tuning %s", MODELS[name].label)
        fitted[name], tuning[name] = fit_and_tune(
            MODELS[name], X["train"], y["train"], X["val"], y["val"], seed=args.seed, tau=tau
        )

    # 4. Evaluate once on test
    metrics = evaluate_models(fitted, y["train"], X["test"], y["test"], horizons, n_boot=args.n_boot, seed=args.seed)
    labels = {n: MODELS[n].label for n in names}
    candidates = [n for n in names if not MODELS[n].baseline] or names
    best = max(candidates, key=lambda n: tuning[n]["val_c_index_uno"])  # chosen on validation, not test

    # Does the best model beat staging alone? Paired bootstrap on the same test patients.
    vs_stage = None
    if "stage" in fitted and best != "stage":
        vs_stage = paired_c_index_difference(
            y["train"],
            y["test"],
            fitted[best].predict(X["test"]),
            fitted["stage"].predict(X["test"]),
            tau=tau,
            n_boot=max(args.n_boot, 1000),
            seed=args.seed,
        )
        log.info(
            "%s vs stage only: dC = %+.3f (95%% CI %.3f to %.3f), p = %.3f",
            best,
            vs_stage["difference"],
            *vs_stage["ci95"],
            vs_stage["p_value"],
        )

    importance = permutation_importance_cindex(fitted[best], X["test"], y["test"], seed=args.seed)

    # 5. Figures + reports
    save_all(fitted, labels, best, y["train"], X["test"], y["test"], horizons, importance, outdir / "figures")
    if is_linear(fitted[best]):
        explain_best(fitted[best], labels[best], X, outdir)
    payload = {
        "best_model": best,
        "horizons_months": horizons.tolist(),
        "split_sizes": split["sizes"],
        "tuning": tuning,
        "best_vs_stage_only": vs_stage,
        "test_metrics": metrics,
        "permutation_importance": [{"feature": f, "mean": m, "std": s} for f, m, s in importance],
    }
    save_metrics(payload, outdir / "metrics.json")
    (outdir / "results.md").write_text(results_table(metrics, labels) + "\n")
    save_model_card(
        outdir / "model_card.md",
        metrics=metrics,
        labels=labels,
        best=best,
        tuning=tuning,
        features=list(X["train"].columns),
        split_sizes=split["sizes"],
        n_rows=len(df),
        n_events=int(df["event"].sum()),
        horizons=horizons,
        importance=importance,
    )

    print("\n" + results_table(metrics, labels) + "\n")
    print(f"Best model: {labels[best]}. Reports written to {outdir}/")
    return payload


if __name__ == "__main__":
    main()
