# tools/generate_publication_figures.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "bc_prepared.csv")
MISSINGNESS_PATH = os.path.join(BASE_DIR, "reports", "tables", "missingness.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

sns.set(style="whitegrid", context="talk")


def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------- FIGURE 2 ---------------- #

def figure2_missing_values_overview():
    """
    Figure 2 – Missing Values Overview (uses real reports/tables/missingness.csv).

    We auto-detect:
      - the "feature" column (name/variable/column)
      - the "missing count" column (contains 'miss' OR first numeric)
    so it works even if the column names differ from my assumptions.
    """
    missing = pd.read_csv(MISSINGNESS_PATH)
    print("Loaded missingness.csv with columns:", list(missing.columns))

    # 1) Guess the feature/name column
    feature_candidates = [
        c for c in missing.columns
        if any(key in c.lower() for key in ["feature", "variable", "column", "name"])
    ]
    if feature_candidates:
        feature_col = feature_candidates[0]
    else:
        # fallback: first non-numeric column, or just the first column
        non_numeric = missing.select_dtypes(exclude=[np.number]).columns
        feature_col = non_numeric[0] if len(non_numeric) > 0 else missing.columns[0]

    # 2) Guess the missing-count column
    count_candidates = [
        c for c in missing.columns
        if ("miss" in c.lower() and "pct" not in c.lower())
    ]
    if count_candidates:
        count_col = count_candidates[0]
    else:
        # fallback: first numeric column that is not obviously a percentage
        numeric_cols = list(missing.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) == 0:
            # last resort: second column
            count_col = missing.columns[1]
        else:
            count_col = numeric_cols[0]

    # Optional: drop rows with 0 missing for a cleaner plot
    missing_plot = missing.copy()
    if count_col in missing_plot.columns:
        missing_plot = missing_plot[missing_plot[count_col] > 0]

    missing_plot = missing_plot.sort_values(count_col, ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=count_col,
        y=feature_col,
        data=missing_plot,
        orient="h",
    )
    plt.xlabel("Number of missing values")
    plt.ylabel("Feature")
    plt.title("Figure 2. Missing Values Overview")
    savefig("Figure2_missing_values_overview.png")


# ---------------- FIGURE 3 ---------------- #

def figure3_duplicate_records_check(df: pd.DataFrame):
    """Figure 3 – Duplicate Records Check (real data/bc_prepared.csv)."""
    dup_mask = df.duplicated()
    n_dup = int(dup_mask.sum())
    n_unique = len(df) - n_dup

    counts = pd.DataFrame({
        "Record type": ["Unique", "Duplicate"],
        "Count": [n_unique, n_dup],
    })

    plt.figure(figsize=(5, 5))
    sns.barplot(x="Record type", y="Count", data=counts)
    plt.title("Figure 3. Duplicate Records Check")
    savefig("Figure3_duplicate_records_check.png")


# ---------------- FIGURE 6 ---------------- #

def figure6_demographic_clinical_distributions(df: pd.DataFrame):
    """
    Figure 6 – Demographic & Clinical Factor Distributions.

    We auto-detect common demographic & clinical columns.
    You can also hardcode the lists if you prefer.
    """
    # Try to auto-detect common demographic fields
    demographic_vars = [
        c for c in df.columns
        if any(k in c.lower() for k in ["sex", "gender", "race", "ethnicity"])
    ]

    # Try to auto-detect common clinical fields
    clinical_vars = [
        c for c in df.columns
        if any(k in c.lower() for k in ["er", "pr", "her2", "stage", "grade", "tumor", "node"])
        and c not in demographic_vars
    ]

    vars_to_plot = demographic_vars + clinical_vars
    if not vars_to_plot:
        print("No demographic/clinical vars auto-detected; "
              "please edit figure6_demographic_clinical_distributions() to set the column list manually.")
        return

    n = len(vars_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for i, col in enumerate(vars_to_plot, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(x=df[col].astype(str))
        plt.xticks(rotation=45, ha="right")
        plt.title(col)

    plt.suptitle("Figure 6. Demographic & Clinical Factor Distributions")
    savefig("Figure6_demographic_clinical_distributions.png")


# ---------------- FIGURE 7 ---------------- #

def figure7_correlation_heatmap(df: pd.DataFrame):
    """Figure 7 – Correlation Heatmap of Numeric Features."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        print("No numeric columns found for correlation heatmap.")
        return

    corr = num_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    plt.title("Figure 7. Correlation Heatmap of Numeric Features")
    savefig("Figure7_correlation_heatmap.png")


# ---------------- MAIN ---------------- #

def main():
    df = pd.read_csv(DATA_PATH)
    print("Loaded bc_prepared.csv with shape:", df.shape)

    figure2_missing_values_overview()
    figure3_duplicate_records_check(df)
    figure6_demographic_clinical_distributions(df)
    figure7_correlation_heatmap(df)


if __name__ == "__main__":
    main()
