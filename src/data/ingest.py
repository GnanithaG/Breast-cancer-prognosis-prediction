# src/data/ingest.py
from __future__ import annotations
import pandas as pd

# columns we try in order
TIME_CANDIDATES = ["time", "Survival Months", "survival_months"]
EVENT_CANDIDATES = ["event", "Status", "vital_status"]


def _coerce_time(df: pd.DataFrame) -> pd.DataFrame:
    """Find a time-like column and standardize it to numeric 'time'."""
    for col in TIME_CANDIDATES:
        if col in df.columns:
            time = df[col]
            break
    else:
        raise ValueError(
            f"Could not find a time column. Looked for {TIME_CANDIDATES}."
        )

    out = df.copy()
    out["time"] = pd.to_numeric(time, errors="coerce")
    if out["time"].isna().all():
        raise ValueError("All values in 'time' are NaN after coercion.")
    return out


def _coerce_event(df: pd.DataFrame) -> pd.DataFrame:
    """Find an event/status-like column and standardize it to binary 'event' (1=death, 0=alive)."""
    for col in EVENT_CANDIDATES:
        if col in df.columns:
            ev = df[col]
            break
    else:
        raise ValueError(
            f"Could not find an event column. Looked for {EVENT_CANDIDATES}."
        )

    # if it is text like "Alive"/"Dead", map it
    if ev.dtype == object:
        ev = (
            ev.astype(str)
              .str.strip()
              .str.lower()
              .map(lambda s: 1 if s.startswith("dead") else 0)
        )

    out = df.copy()
    out["event"] = pd.to_numeric(ev, errors="coerce").fillna(0).astype(int)
    return out


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV and ensure it contains standardized columns:
      - 'time': numeric follow-up duration
      - 'event': 1 = event/death, 0 = censored/alive

    This is flexible with common Kaggle breast-cancer/SEER-style names.
    """
    df = pd.read_csv(path)

    # standardize time and event
    df = _coerce_time(df)
    df = _coerce_event(df)

    # sanity checks
    if (df["time"] <= 0).all():
        raise ValueError("All 'time' values are non-positive; check the source column.")
    if not set(df["event"].unique()).issubset({0, 1}):
        raise ValueError("'event' must be binary (0/1) after coercion.")

    # nice ordering: time, event, then everything else
    other_cols = [c for c in df.columns if c not in ("time", "event")]
    df = df[["time", "event", *other_cols]]

    return df
