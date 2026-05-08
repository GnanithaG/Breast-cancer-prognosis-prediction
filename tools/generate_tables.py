#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def _safe_list(x):
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return list(x)
    return [x]

def _flatten_metrics(metrics_dict):
    """
    Expect a structure like:
      {
        "rsf": {
          "by_horizon": {
             "12": {"cindex": 0.82, "brier": 0.12},
             "36": {...}
          },
          "aggregate": {"ibs": 0.14}
        },
        "cox": {...}
      }

    But we don't assume exact keys. We try to pull per-horizon rows and aggregate rows
    for each model, with whatever metrics exist.
    """
    rows_hz = []
    rows_aggr = []
    for model, block in metrics_dict.items():
        if not isinstance(block, dict):
            # Ignore stray keys
            continue

        # Per-horizon section (common names we try)
        per_hz_candidates = ["by_horizon", "per_horizon", "horizons", "timepoints"]
        per_hz = None
        for k in per_hz_candidates:
            if k in block and isinstance(block[k], dict):
                per_hz = block[k]
                break

        if per_hz is not None:
            for h, vals in per_hz.items():
                # normalize horizon to int if possible
                try:
                    horizon = int(float(h))
                except Exception:
                    horizon = h
                row = {"model": model, "horizon": horizon}
                if isinstance(vals, dict):
                    for mk, mv in vals.items():
                        row[str(mk).lower()] = mv
                rows_hz.append(row)

        # Aggregate section (common names we try)
        aggr_candidates = ["aggregate", "overall", "summary"]
        aggr = None
        for k in aggr_candidates:
            if k in block and isinstance(block[k], dict):
                aggr = block[k]
                break
        if aggr is not None:
            row = {"model": model}
            for mk, mv in aggr.items():
                row[str(mk).lower()] = mv
            rows_aggr.append(row)

    df_hz = pd.DataFrame(rows_hz).sort_values(["model", "horizon"], ignore_index=True) if rows_hz else pd.DataFrame()
    df_aggr = pd.DataFrame(rows_aggr).sort_values(["model"], ignore_index=True) if rows_aggr else pd.DataFrame()
    return df_hz, df_aggr

def _mk_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_tables(metrics_path, data_path, outdir, make_latex=False):
    outdir = Path(outdir)
    _mk_outdir(outdir)

    # 1) METRICS TABLES
    if metrics_path and Path(metrics_path).exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        df_hz, df_aggr = _flatten_metrics(metrics)

        if not df_hz.empty:
            # Wide-by-metric & model as rows for each horizon
            df_hz.to_csv(outdir / "metrics_by_model_horizon.csv", index=False)
            # Split out common metrics if present
            for metric_name in ["cindex", "brier", "ibs", "auc", "ece"]:
                cols = [c for c in df_hz.columns if metric_name in c or c == metric_name]
                if cols:
                    tab = df_hz[["model", "horizon"] + cols].copy()
                    tab.sort_values(["horizon", "model"], inplace=True)
                    tab.to_csv(outdir / f"{metric_name}_by_model_horizon.csv", index=False)
                    with open(outdir / f"{metric_name}_by_model_horizon.md", "w") as md:
                        md.write(tab.to_markdown(index=False))
                    if make_latex:
                        with open(outdir / f"{metric_name}_by_model_horizon.tex", "w") as tex:
                            tex.write(tab.to_latex(index=False, float_format="%.4f"))

        if not df_aggr.empty:
            df_aggr.to_csv(outdir / "metrics_aggregate_by_model.csv", index=False)
            with open(outdir / "metrics_aggregate_by_model.md", "w") as md:
                md.write(df_aggr.to_markdown(index=False))
            if make_latex:
                with open(outdir / "metrics_aggregate_by_model.tex", "w") as tex:
                    tex.write(df_aggr.to_latex(index=False, float_format="%.4f"))

    # 2) DATASET PROFILE TABLES
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)

        # Basic counts
        basic = pd.DataFrame({
            "rows": [len(df)],
            "columns": [len(df.columns)],
            "generated_at": [datetime.now().isoformat(timespec="seconds")]
        })
        basic.to_csv(outdir / "dataset_basic.csv", index=False)
        with open(outdir / "dataset_basic.md", "w") as md:
            md.write(basic.to_markdown(index=False))

        # Outcome balance if columns exist
        event_col = None
        time_col = None
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ("event", "status", "vital_status"):
                event_col = c
            if cl in ("time", "survival_months", "survival time", "duration"):
                time_col = c

        if event_col:
            bal = df[event_col].value_counts(dropna=False).rename_axis("event").reset_index(name="n")
            bal["event"] = bal["event"].astype(str)
            bal["pct"] = (bal["n"] / len(df) * 100).round(2)
            bal.to_csv(outdir / "outcome_balance.csv", index=False)
            with open(outdir / "outcome_balance.md", "w") as md:
                md.write(bal.to_markdown(index=False))
            if make_latex:
                with open(outdir / "outcome_balance.tex", "w") as tex:
                    tex.write(bal.to_latex(index=False))

        # Numeric summary
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            desc = df[num_cols].describe(percentiles=[0.25,0.5,0.75]).T
            desc = desc.rename(columns={
                "50%": "median",
                "25%": "q1",
                "75%": "q3"
            })
            desc.to_csv(outdir / "numeric_description.csv")
            with open(outdir / "numeric_description.md", "w") as md:
                md.write(desc.to_markdown())

        # Missingness table
        miss = (df.isna().sum()).rename("missing").to_frame()
        miss["missing_pct"] = (miss["missing"] / len(df) * 100).round(2)
        miss = miss.sort_values("missing", ascending=False)
        miss.to_csv(outdir / "missingness.csv")
        with open(outdir / "missingness.md", "w") as md:
            md.write(miss.to_markdown())

def main():
    ap = argparse.ArgumentParser(description="Generate publication-ready tables from metrics.json and dataset.")
    ap.add_argument("--metrics", default="reports/metrics.json", help="Path to metrics.json")
    ap.add_argument("--data", default="data/bc_prepared.csv", help="Path to the dataset used")
    ap.add_argument("--outdir", default="reports/tables", help="Output directory for tables")
    ap.add_argument("--latex", action="store_true", help="Also emit LaTeX tables")
    args = ap.parse_args()

    make_tables(args.metrics, args.data, args.outdir, make_latex=args.latex)
    print(f"[tables] Saved to {args.outdir}")

if __name__ == "__main__":
    main()
