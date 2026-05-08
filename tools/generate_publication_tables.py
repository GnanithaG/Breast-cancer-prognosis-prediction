# tools/generate_publication_tables.py

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "bc_prepared.csv")
TABLE_SRC_DIR = os.path.join(BASE_DIR, "reports", "tables")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
CONFIG_SPLITS = os.path.join(BASE_DIR, "configs", "split_indices.json")
OUT_DIR = TABLE_SRC_DIR  # reuse reports/tables for final tables too

os.makedirs(OUT_DIR, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    md_path = os.path.join(OUT_DIR, f"{name}.md")

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    print(f"Saved {csv_path} and {md_path}")


# ---------------- TABLE 1: Cohort Summary ---------------- #

def table1_cohort_summary(df: pd.DataFrame):
    """
    Table 1. Cohort Summary (counts, medians, IQRs).

    We’ll combine basic info from:
      - outcome column (auto-detect)
      - follow-up / time column (auto-detect)
      - numeric_description.csv if needed
    """

    # Guess event column: look for "event" / "status" / "outcome"
    event_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["event", "status", "outcome", "death"])
    ]
    event_col = event_candidates[0] if event_candidates else None

    # Guess time column: look for "time" / "month" / "follow"
    time_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["time", "month", "follow"])
    ]
    time_col = time_candidates[0] if time_candidates else None

    summary_rows = []

    summary_rows.append({"Measure": "N (patients)", "Value": len(df)})

    if event_col is not None:
        events = int(df[event_col].sum())
        cens = int(len(df) - events)
        summary_rows.append({"Measure": "Events (deaths)", "Value": events})
        summary_rows.append({"Measure": "Censored", "Value": cens})

    if time_col is not None:
        med = df[time_col].median()
        q1 = df[time_col].quantile(0.25)
        q3 = df[time_col].quantile(0.75)
        summary_rows.append({
            "Measure": f"Median follow-up ({time_col})",
            "Value": round(med, 2),
        })
        summary_rows.append({
            "Measure": f"IQR follow-up ({time_col})",
            "Value": f"{round(q1, 2)}–{round(q3, 2)}",
        })

    # Optionally, incorporate age & tumor if present
    for col_key in ["age", "tumor"]:
        candidates = [c for c in df.columns if col_key in c.lower()]
        if candidates:
            col = candidates[0]
            med = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            summary_rows.append({
                "Measure": f"Median {col}",
                "Value": round(med, 2),
            })
            summary_rows.append({
                "Measure": f"IQR {col}",
                "Value": f"{round(q1, 2)}–{round(q3, 2)}",
            })

    table = pd.DataFrame(summary_rows)
    save_table(table, "Table1_cohort_summary")


# ---------------- TABLE 2: Event Balance per Split ---------------- #

def table2_event_balance_per_split(df: pd.DataFrame):
    """
    Table 2. Event Balance per Split (train/val/test).
    We try to find a 'split' column; if there is none, we fall back to split_indices.json.
    """

    # Try split column first
    split_candidates = [c for c in df.columns if "split" in c.lower()]
    if split_candidates:
        split_col = split_candidates[0]
        print("Using split column:", split_col)

        # Guess event column
        event_candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ["event", "status", "outcome", "death"])
        ]
        event_col = event_candidates[0] if event_candidates else None

        rows = []
        for split, sub in df.groupby(split_col):
            n = len(sub)
            if event_col is not None:
                events = int(sub[event_col].sum())
                cens = n - events
                events_pct = events / n * 100 if n > 0 else np.nan
            else:
                events = np.nan
                cens = np.nan
                events_pct = np.nan
            rows.append({
                "Split": split,
                "N": n,
                "Events": events,
                "Censored": cens,
                "Event %": round(events_pct, 1) if not np.isnan(events_pct) else "",
            })

        table = pd.DataFrame(rows)
        save_table(table, "Table2_event_balance_per_split")
        return

    # Fallback: use split_indices.json
    if not os.path.exists(CONFIG_SPLITS):
        print("No split column and no split_indices.json found – "
              "cannot build Table 2 automatically.")
        return

    with open(CONFIG_SPLITS, "r") as f:
        split_cfg = json.load(f)

    # Expect structure like {"train": [idx...], "val": [...], "test": [...]}
    event_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["event", "status", "outcome", "death"])
    ]
    event_col = event_candidates[0] if event_candidates else None

    rows = []
    for split_name in ["train", "val", "test"]:
       indices = split_cfg[split_name]
       sub = df.iloc[indices]
        n = len(sub)
        if event_col is not None:
            events = int(sub[event_col].sum())
            cens = n - events
            events_pct = events / n * 100 if n > 0 else np.nan
        else:
            events = np.nan
            cens = np.nan
            events_pct = np.nan
        rows.append({
            "Split": split_name,
            "N": n,
            "Events": events,
            "Censored": cens,
            "Event %": round(events_pct, 1) if not np.isnan(events_pct) else "",
        })

    table = pd.DataFrame(rows)
    save_table(table, "Table2_event_balance_per_split")


# ---------------- TABLE 3: Model Metrics by Horizon and Model ---------------- #

def table3_model_metrics():
    """
    Table 3. Model Metrics by Horizon and Model.
    This depends on your metrics.json structure.
    We'll assume something like:

    {
      "RSF": {
          "12": {"c_index": 0.78, "brier": 0.09},
          "36": {...},
          ...
      },
      "Cox": {...}
    }

    Adjust the parsing if your format is different.
    """

    if not os.path.exists(METRICS_PATH):
        print("metrics.json not found; cannot build Table 3 automatically.")
        return

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    rows = []

    for model_name, horizons in metrics.items():
        # horizons might be nested; we try to iterate levels that look like horizons
        for horizon_key, vals in horizons.items():
            # Try to interpret horizon
            try:
                horizon = int(horizon_key)
            except ValueError:
                # skip non-horizon keys
                continue

            c_index = None
            brier = None
            ibs = None

            # auto-detect metric keys
            for k, v in vals.items():
                lk = k.lower()
                if "cindex" in lk or "c_index" in lk or "concord" in lk:
                    c_index = v
                elif "brier" in lk and "integrated" not in lk:
                    brier = v
                elif "ibs" in lk or "integrated_brier" in lk:
                    ibs = v

            rows.append({
                "Horizon (months)": horizon,
                "Model": model_name,
                "C-index": round(c_index, 3) if c_index is not None else "",
                "IBS": round(ibs, 3) if ibs is not None else "",
                "Brier score": round(brier, 3) if brier is not None else "",
            })

    if not rows:
        print("No usable entries parsed from metrics.json; "
              "please adjust table3_model_metrics() to your file format.")
        return

    table = pd.DataFrame(rows).sort_values(["Model", "Horizon (months)"])
    save_table(table, "Table3_model_metrics")


# ---------------- TABLE 4: Feature Catalog ---------------- #

def table4_feature_catalog(df: pd.DataFrame):
    """
    Table 4. Feature Catalog (name, type, encoding).
    For now we infer type by dtype and set encoding heuristically.
    You can hand-edit the CSV later if you want to refine encodings.
    """

    rows = []
    for col in df.columns:
        # Skip obvious non-feature columns if needed
        if any(k in col.lower() for k in ["id", "index"]):
            continue

        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            ftype = "numeric"
            encoding = "scaled / numeric"
        else:
            ftype = "categorical"
            encoding = "one-hot / categorical"

        rows.append({
            "Feature name": col,
            "Type": ftype,
            "Original dtype": dtype,
            "Encoding": encoding,
        })

    table = pd.DataFrame(rows)
    save_table(table, "Table4_feature_catalog")


# ---------------- TABLE 5: Train/Val/Test Sizes and Seed ---------------- #

def table5_split_sizes(df: pd.DataFrame):
    """
    Table 5. Train/Validation/Test Sizes and Seed.
    Seed is taken from split_indices.json if present; otherwise left blank.
    """

    seed = ""
    if os.path.exists(CONFIG_SPLITS):
        with open(CONFIG_SPLITS, "r") as f:
            split_cfg = json.load(f)
        if isinstance(split_cfg, dict) and "seed" in split_cfg:
            seed = split_cfg["seed"]

    split_candidates = [c for c in df.columns if "split" in c.lower()]
    if split_candidates:
        split_col = split_candidates[0]
        rows = []
        for split, sub in df.groupby(split_col):
            rows.append({
                "Split": split,
                "N": len(sub),
                "Random seed": seed,
            })
        table = pd.DataFrame(rows)
        save_table(table, "Table5_split_sizes_and_seed")
        return

    # Fallback: use split_indices.json if it has indices only
    if os.path.exists(CONFIG_SPLITS):
        with open(CONFIG_SPLITS, "r") as f:
            split_cfg = json.load(f)
        if isinstance(split_cfg, dict):
            rows = []
            for split_name, indices in split_cfg.items():
                if split_name == "seed":
                    continue
                if isinstance(indices, list):
                    n = len(indices)
                else:
                    n = len(df.loc[indices])
                rows.append({
                    "Split": split_name,
                    "N": n,
                    "Random seed": seed,
                })
            table = pd.DataFrame(rows)
            save_table(table, "Table5_split_sizes_and_seed")
            return

    print("Could not determine split sizes; please adjust table5_split_sizes().")


# ---------------- MAIN ---------------- #

def main():
    df = pd.read_csv(DATA_PATH)
    print("Loaded bc_prepared.csv with shape:", df.shape)

    table1_cohort_summary(df)
    table2_event_balance_per_split(df)
    table3_model_metrics()
    table4_feature_catalog(df)
    table5_split_sizes(df)


if __name__ == "__main__":
    main()
