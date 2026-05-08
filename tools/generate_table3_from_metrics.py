# tools/generate_table3_from_metrics.py

import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
OUT_DIR = os.path.join(BASE_DIR, "reports", "tables")
os.makedirs(OUT_DIR, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    md_path = os.path.join(OUT_DIR, f"{name}.md")
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    print(f"Saved {csv_path} and {md_path}")


def main():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"metrics.json not found at {METRICS_PATH}")

    with open(METRICS_PATH, "r") as f:
        data = json.load(f)

    # We expect keys "cox" and "rsf", each with:
    # c_index_ipcw, ibs, t_auc_times, t_auc, brier_times, brier_scores
    rows = []

    for model_key, m in data.items():
        if not isinstance(m, dict):
            continue

        model_name = model_key.upper()  # "COX", "RSF"
        # Make nicer labels
        if model_name == "COX":
            model_name = "Cox"
        elif model_name == "RSF":
            model_name = "RSF"

        c_index = m.get("c_index_ipcw")
        ibs = m.get("ibs")

        t_times = m.get("t_auc_times", [])
        t_aucs = m.get("t_auc", [])
        b_times = m.get("brier_times", [])
        b_scores = m.get("brier_scores", [])

        # Sanity: ensure lists are same length as times
        max_len = max(len(t_times), len(t_aucs), len(b_times), len(b_scores))

        for i in range(max_len):
            horizon = None
            if i < len(t_times):
                horizon = t_times[i]
            elif i < len(b_times):
                horizon = b_times[i]

            if horizon is None:
                continue

            auc_val = t_aucs[i] if i < len(t_aucs) else None
            brier_val = b_scores[i] if i < len(b_scores) else None

            row = {
                "Horizon (months)": int(horizon),
                "Model": model_name,
                "C-index": round(float(c_index), 3) if c_index is not None else "",
                "IBS": round(float(ibs), 3) if ibs is not None else "",
                "Time-dependent AUC": (
                    round(float(auc_val), 3) if auc_val is not None else ""
                ),
                "Brier score": (
                    round(float(brier_val), 3) if brier_val is not None else ""
                ),
            }
            rows.append(row)

    if not rows:
        print("No rows could be constructed from metrics.json – please inspect file.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["Model", "Horizon (months)"])
    save_table(df, "Table3_model_metrics")


if __name__ == "__main__":
    main()
