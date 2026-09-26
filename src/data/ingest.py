"""Load the raw SEER-style breast cancer CSV into a clean modelling table.

The output always has:
  * ``time``  - follow-up in months (float, > 0)
  * ``event`` - 1 = death observed, 0 = censored (alive at last follow-up)
  * snake_case feature columns

The raw outcome columns (e.g. ``Survival Months`` / ``Status``) are *removed*
so they can never leak into the feature matrix.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

TIME_CANDIDATES = ("time", "Survival Months", "survival_months")
EVENT_CANDIDATES = ("event", "Status", "vital_status")

# Words that mean "the event (death) happened".
_EVENT_TRUE = {"dead", "deceased", "died", "death", "1", "true", "yes"}
_EVENT_FALSE = {"alive", "censored", "living", "0", "false", "no"}

# Fix known typos / awkward names in the public Kaggle SEER extract.
_RENAME = {
    "reginol_node_positive": "regional_node_positive",
    "6th_stage": "stage_6th",
    "differentiate": "differentiation",
}


def to_snake(name: str) -> str:
    """'T Stage ' -> 't_stage', 'Regional Node Examined' -> 'regional_node_examined'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return _RENAME.get(s, s)


def _find(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a {what} column. Looked for {list(candidates)}.")


def coerce_event(values: pd.Series) -> pd.Series:
    """Map text or numeric status values to 0/1, failing loudly on anything unknown.

    Works regardless of the pandas string dtype (object, ``str`` or ``string``).
    """
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        out = pd.to_numeric(values, errors="coerce")
    else:
        norm = values.astype("string").str.strip().str.lower()
        out = norm.map(lambda s: 1 if s in _EVENT_TRUE else (0 if s in _EVENT_FALSE else None))
    bad = out.isna() | ~out.isin([0, 1])
    if bad.any():
        examples = values[bad].astype(str).unique()[:5].tolist()
        raise ValueError(f"Unrecognised event/status values: {examples}")
    return out.astype(int)


def load_csv(path: str, drop_duplicates: bool = True) -> pd.DataFrame:
    """Read ``path`` and return a clean frame with ``time``, ``event`` + features."""
    raw = pd.read_csv(path)
    time_src = _find(raw, TIME_CANDIDATES, "time")
    event_src = _find(raw, EVENT_CANDIDATES, "event")

    time = pd.to_numeric(raw[time_src], errors="coerce")
    event = coerce_event(raw[event_src])

    features = raw.drop(columns=[time_src, event_src])
    # Remove any *other* outcome-like columns too, so nothing leaks.
    features = features.drop(columns=[c for c in features.columns if c in TIME_CANDIDATES + EVENT_CANDIDATES])
    features.columns = [to_snake(c) for c in features.columns]

    df = pd.concat([time.rename("time"), event.rename("event"), features], axis=1)

    n0 = len(df)
    df = df[df["time"].notna() & (df["time"] > 0)]
    if len(df) < n0:
        log.warning("Dropped %d rows with missing or non-positive follow-up time", n0 - len(df))

    if drop_duplicates:
        n1 = len(df)
        df = df.drop_duplicates()
        if len(df) < n1:
            log.info("Dropped %d exact duplicate rows", n1 - len(df))

    if df["event"].sum() == 0:
        raise ValueError("No events in the data - every patient is censored.")

    return df.reset_index(drop=True)
