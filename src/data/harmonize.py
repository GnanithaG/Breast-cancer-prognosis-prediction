"""Clean up clinical coding so categories are consistent.

Written against the column names produced by :func:`src.data.ingest.load_csv`
(snake_case). Anything not present is simply skipped.
"""

from __future__ import annotations

import pandas as pd

RECEPTOR_COLS = ("estrogen_status", "progesterone_status", "er_status", "pr_status", "her2_status")
NON_NEGATIVE_COLS = ("age", "tumor_size", "regional_node_examined", "regional_node_positive")


def norm_receptor(v) -> str:
    if pd.isna(v):
        return "Unknown"
    v = str(v).strip().lower()
    if v in {"pos", "positive", "1", "true", "yes", "+"}:
        return "Positive"
    if v in {"neg", "negative", "0", "false", "no", "-"}:
        return "Negative"
    return "Unknown"


def norm_grade(g) -> str:
    """'3' -> '3', 'Grade II' -> '2', ' anaplastic; Grade IV' -> '4'."""
    if pd.isna(g):
        return "Unknown"
    s = str(g).strip().upper()
    if "ANAPLASTIC" in s:
        return "4"
    s = s.replace("GRADE", "").strip(" ;")
    roman = {"I": "1", "II": "2", "III": "3", "IV": "4"}
    if s in roman:
        return roman[s]
    if s in {"1", "2", "3", "4"}:
        return s
    return "Unknown"


def harmonize_registry_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trim stray whitespace in every text column.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().astype(object)

    for col in RECEPTOR_COLS:
        if col in df.columns:
            df[col] = df[col].map(norm_receptor)

    if "grade" in df.columns:
        df["grade"] = df["grade"].map(norm_grade)

    for col in NON_NEGATIVE_COLS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = float("nan")

    # Positive nodes can't exceed nodes examined.
    if {"regional_node_positive", "regional_node_examined"} <= set(df.columns):
        df["regional_node_positive"] = df[["regional_node_positive", "regional_node_examined"]].min(axis=1)

    return df
