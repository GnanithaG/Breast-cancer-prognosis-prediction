"""Frozen, stratified train / validation / test splits.

Indices are written to JSON together with a fingerprint of the data, so a later
run on a *different* file (or a re-cleaned file) regenerates the split instead of
silently reusing row numbers that now point at different patients.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def data_fingerprint(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def make_or_load_splits(
    df: pd.DataFrame,
    save_path: Path,
    event_col: str = "event",
    stratify_col: str | None = "stage_6th",
    seed: int = 7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict:
    save_path = Path(save_path)
    fp = data_fingerprint(df)

    if save_path.exists():
        saved = json.loads(save_path.read_text())
        if saved.get("fingerprint") == fp:
            return saved
        log.warning("Data changed since %s was written - regenerating splits.", save_path)

    strat = df[event_col].astype(str)
    if stratify_col and stratify_col in df.columns:
        strat = df[stratify_col].fillna("Unknown").astype(str) + "_" + strat
    # Merge strata too small to split into a catch-all bucket.
    counts = strat.value_counts()
    strat = strat.where(~strat.isin(counts[counts < 10].index), "rare_" + df[event_col].astype(str))

    idx = df.index.to_numpy()
    trn, tmp = train_test_split(idx, test_size=val_size + test_size, stratify=strat, random_state=seed)
    val, tst = train_test_split(
        tmp, test_size=test_size / (val_size + test_size), stratify=strat.loc[tmp], random_state=seed
    )

    result = {
        "fingerprint": fp,
        "seed": seed,
        "stratified_on": [c for c in (stratify_col, event_col) if c and c in df.columns],
        "sizes": {"train": len(trn), "val": len(val), "test": len(tst)},
        "train": sorted(int(i) for i in trn),
        "val": sorted(int(i) for i in val),
        "test": sorted(int(i) for i in tst),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    return result
