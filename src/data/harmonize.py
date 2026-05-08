import pandas as pd

def _norm_stage(s):
    if pd.isna(s): return "Unknown"
    s = str(s).strip().upper().replace("STAGE ", "")
    if s.startswith("IIB"):  # collapse all finer T/N into broad stages
        return "II"
    if s.startswith("IIIB") or s.startswith("IIIC"):
        return "III"
    if s.startswith("II"): return "II"
    if s.startswith("III"): return "III"
    if s.startswith("IV"): return "IV"
    if s.startswith("I"): return "I"
    return "Unknown"

def _norm_receptor(v):
    if pd.isna(v): return "Unknown"
    v = str(v).strip().lower()
    if v in ("pos","positive","1","true","yes"): return "Positive"
    if v in ("neg","negative","0","false","no"): return "Negative"
    return "Unknown"

def _norm_grade(g):
    if pd.isna(g): return "Unknown"
    g = str(g).strip().upper().replace("GRADE", "").strip()
    if g in {"1","2","3","4"}: return g
    if g in {"I","II","III","IV"}:
        return {"I":"1","II":"2","III":"3","IV":"4"}[g]
    return "Unknown"

def harmonize_registry_codes(df: pd.DataFrame, stage_col: str = "stage") -> pd.DataFrame:
    df = df.copy()
    if stage_col in df.columns:
        df[stage_col] = df[stage_col].map(_norm_stage)
    for col in ("er_status","pr_status","her2_status"):
        if col in df.columns:
            df[col] = df[col].map(_norm_receptor)
    if "grade" in df.columns:
        df["grade"] = df["grade"].map(_norm_grade)
    # sanitize numeric non-negativity for common fields
    for num_col in ("age","tumor_size","nodes_positive","nodes_examined","survival_months"):
        if num_col in df.columns:
            df.loc[df[num_col] < 0, num_col] = None
    # keep nodes_positive <= nodes_examined
    if "nodes_positive" in df.columns and "nodes_examined" in df.columns:
        df["nodes_positive"] = df[["nodes_positive","nodes_examined"]].min(axis=1)
    return df
