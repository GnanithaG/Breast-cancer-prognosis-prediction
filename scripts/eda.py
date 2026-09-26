#!/usr/bin/env python
"""Cohort description: Table 1, missingness, and exploratory figures.

Replaces the old tools/generate_*.py scripts with one script that reads the
same cleaned data the models use.

    python scripts/eda.py            # writes to reports/eda/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402

from src.data.harmonize import harmonize_registry_codes  # noqa: E402
from src.data.ingest import load_csv  # noqa: E402
from src.evaluation import plots  # noqa: E402,F401  (applies shared matplotlib style)

COLORS = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]


def table1(df: pd.DataFrame) -> pd.DataFrame:
    """Characteristics overall and by outcome (Alive / Dead)."""
    groups = {"Overall": df, "Alive": df[df.event == 0], "Died": df[df.event == 1]}
    rows = [{"Variable": "Patients, n", **{k: f"{len(g)}" for k, g in groups.items()}}]
    rows.append(
        {
            "Variable": "Follow-up months, median [IQR]",
            **{
                k: f"{g.time.median():.0f} [{g.time.quantile(0.25):.0f}-{g.time.quantile(0.75):.0f}]"
                for k, g in groups.items()
            },
        }
    )
    feats = df.drop(columns=["time", "event"])
    for col in feats.columns:
        if pd.api.types.is_numeric_dtype(feats[col]):
            rows.append(
                {
                    "Variable": f"{col}, median [IQR]",
                    **{
                        k: f"{g[col].median():.0f} [{g[col].quantile(0.25):.0f}-{g[col].quantile(0.75):.0f}]"
                        for k, g in groups.items()
                    },
                }
            )
        else:
            rows.append({"Variable": f"{col}, n (%)", **{k: "" for k in groups}})
            for level in df[col].value_counts().index:
                rows.append(
                    {
                        "Variable": f"&emsp;{level}",
                        **{k: f"{(g[col] == level).sum()} ({(g[col] == level).mean():.1%})" for k, g in groups.items()},
                    }
                )
    return pd.DataFrame(rows)


def fig_outcome(df, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    counts = df.event.map({0: "Alive / censored", 1: "Died"}).value_counts()
    a1.bar(counts.index, counts.values, color=[COLORS[0], COLORS[3]])
    for i, v in enumerate(counts.values):
        a1.text(i, v, f"{v} ({v / len(df):.0%})", ha="center", va="bottom")
    a1.set(title="Outcome", ylabel="Patients")
    a1.grid(axis="x", visible=False)
    bins = np.arange(0, df.time.max() + 6, 6)
    a2.hist(df.time[df.event == 0], bins=bins, color=COLORS[0], alpha=0.7, label="Censored")
    a2.hist(df.time[df.event == 1], bins=bins, color=COLORS[3], alpha=0.8, label="Died")
    a2.set(title="Follow-up time", xlabel="Months", ylabel="Patients")
    a2.legend()
    fig.savefig(path)
    plt.close(fig)


def fig_km(df, path, by="stage_6th"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    t, s, ci = kaplan_meier_estimator(df.event.astype(bool), df.time, conf_type="log-log")
    a1.step(t, s, where="post", color=COLORS[0])
    a1.fill_between(t, ci[0], ci[1], step="post", alpha=0.15, color=COLORS[0])
    a1.set(title="Overall survival (Kaplan-Meier)", xlabel="Months", ylabel="Survival probability", ylim=(0, 1.02))
    for i, level in enumerate(sorted(df[by].dropna().unique())):
        m = df[by] == level
        t, s = kaplan_meier_estimator(df.event[m].astype(bool), df.time[m])
        a2.step(t, s, where="post", color=COLORS[i % len(COLORS)], label=f"{level} (n={m.sum()})")
    a2.set(title=f"By {by}", xlabel="Months")
    a2.legend(fontsize=8, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def fig_numeric(df, path):
    num = df.drop(columns=["time", "event"]).select_dtypes("number").columns
    fig, axes = plt.subplots(1, len(num), figsize=(3.2 * len(num), 3.2))
    for ax, col in zip(np.atleast_1d(axes), num):
        data = [df.loc[df.event == e, col].dropna() for e in (0, 1)]
        ax.boxplot(data, tick_labels=["Alive", "Died"], widths=0.5, showfliers=False)
        ax.set(title=col)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Numeric features by outcome", y=1.03)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/Breast_Cancer.csv")
    p.add_argument("--outdir", default="reports/eda")
    args = p.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    df = harmonize_registry_codes(load_csv(args.data))

    t1 = table1(df)
    (out / "table1_cohort.md").write_text(t1.to_markdown(index=False) + "\n")
    t1.to_csv(out / "table1_cohort.csv", index=False)

    miss = pd.DataFrame({"missing": raw.isna().sum(), "missing_pct": (raw.isna().mean() * 100).round(2)}).rename_axis(
        "column"
    )
    quality = [
        f"- Raw rows: {len(raw)}; exact duplicate rows removed: {int(raw.duplicated().sum())}",
        f"- Columns with any missing values: {int((miss.missing > 0).sum())}",
    ]
    (out / "data_quality.md").write_text("# Data quality\n\n" + "\n".join(quality) + "\n\n" + miss.to_markdown() + "\n")

    fig_outcome(df, out / "outcome_and_followup.png")
    fig_km(df, out / "kaplan_meier.png")
    fig_numeric(df, out / "numeric_by_outcome.png")
    print(f"EDA written to {out}/")


if __name__ == "__main__":
    main()
