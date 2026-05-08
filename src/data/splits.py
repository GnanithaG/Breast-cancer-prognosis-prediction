# src/data/splits.py
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def make_or_load_splits(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    stage_col: str,
    save_path: Path,
    seed: int = 7,
):
    """
    Create (or load) frozen indices for train/val/test.

    Strategy:
    - stratify on a combination of clinical stage (if present) and event,
      so each split has similar outcome distribution and stage mix.
    - 70% train
    - remaining 30% is split 50/50 into val and test (so 15/15)
    - write to JSON so future runs use the exact same rows
    """
    # if we already created them once, just reuse
    if save_path.exists():
        return json.loads(save_path.read_text())

    # build stratification label
    if stage_col in df.columns:
        stage = df[stage_col].fillna("Unknown").astype(str)
    else:
        stage = pd.Series(["All"] * len(df), index=df.index)

    event = df[event_col].astype(int)
    strat = stage.astype(str) + "_" + event.astype(str)

    # first split: 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    trn_idx, tmp_idx = next(sss1.split(df, strat))

    # second split: temp -> 50% val, 50% test (i.e. 15% / 15% of original)
    tmp = df.iloc[tmp_idx]
    strat_tmp = strat.iloc[tmp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, tst_rel = next(sss2.split(tmp, strat_tmp))
    val_idx = tmp.index[val_rel].to_list()
    tst_idx = tmp.index[tst_rel].to_list()

    result = {
        "train": df.index[trn_idx].to_list(),
        "val": val_idx,
        "test": tst_idx,
        "seed": seed,
        "stratified_on": [event_col, stage_col],
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
    }

    # save so all future runs are identical
    save_path.write_text(json.dumps(result, indent=2))
    return result
