"""Everything the demo app needs, built from the repo's own data, split and tuned settings.

The app refits the tuned Cox model on the training split at start-up (about a second),
instead of loading a pickled model that could break across scikit-learn versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from src.data.harmonize import harmonize_registry_codes
from src.data.ingest import load_csv
from src.data.splits import make_or_load_splits
from src.evaluation.explain import linear_shap
from src.preprocessing.pipeline import build_preprocessor

GRADE_TO_DIFFERENTIATION = {
    "1": "Well differentiated",
    "2": "Moderately differentiated",
    "3": "Poorly differentiated",
    "4": "Undifferentiated",
}


def stage_from_tn(t: str, n: str) -> str:
    """AJCC 6th-edition stage group for M0 disease from T and N (matches the data exactly)."""
    if n == "N3":
        return "IIIC"
    if t == "T4":
        return "IIIB"
    if n == "N2" or t == "T3":
        return "IIIA"
    return {"T1": "IIA", "T2": "IIB"}[t]


@dataclass
class ServingModel:
    model: Pipeline
    X_train: pd.DataFrame
    km_time: np.ndarray
    km_surv: np.ndarray
    risk_cutoffs: np.ndarray  # quartiles of training risk scores
    test_c_index: float
    alpha: float

    @property
    def schema(self) -> dict:
        out = {}
        for col in self.X_train.columns:
            s = self.X_train[col]
            if pd.api.types.is_numeric_dtype(s):
                out[col] = {"type": "numeric", "min": int(s.min()), "max": int(s.max()), "median": int(s.median())}
            else:
                out[col] = {"type": "categorical", "levels": s.value_counts().index.tolist()}
        return out

    def complete(self, inputs: dict) -> pd.DataFrame:
        """Fill in the derived columns (stage group, differentiation) so they stay consistent."""
        row = dict(inputs)
        row["stage_6th"] = stage_from_tn(row["t_stage"], row["n_stage"])
        if "differentiation" in self.X_train.columns:
            row["differentiation"] = GRADE_TO_DIFFERENTIATION[str(row["grade"])]
        return pd.DataFrame([row])[self.X_train.columns]

    def predict(self, X: pd.DataFrame, times) -> dict:
        fn = self.model.predict_survival_function(X)[0]
        lo, hi = fn.domain
        t = np.clip(np.asarray(times, float), lo, hi)
        risk = float(self.model.predict(X)[0])
        group = int(np.searchsorted(self.risk_cutoffs, risk))  # 0..3
        return {
            "times": t,
            "survival": fn(t),
            "risk_score": risk,
            "risk_group": ["Lowest", "Low-mid", "High-mid", "Highest"][group],
            "risk_quartile": group + 1,
            "shap": linear_shap(self.model, X, self.X_train).iloc[0],
        }

    def population_survival(self, times) -> np.ndarray:
        idx = np.searchsorted(self.km_time, times, side="right") - 1
        return np.where(idx >= 0, self.km_surv[np.clip(idx, 0, None)], 1.0)


def build_serving_model(
    data="data/Breast_Cancer.csv", splits="configs/split_indices.json", metrics="reports/metrics.json"
) -> ServingModel:
    df = harmonize_registry_codes(load_csv(data))
    split = make_or_load_splits(df, Path(splits))
    trn, tst = df.loc[split["train"]], df.loc[split["test"]]
    X_train, X_test = trn.drop(columns=["time", "event"]), tst.drop(columns=["time", "event"])
    y_train = Surv.from_arrays(trn["event"].astype(bool), trn["time"].astype(float))
    y_test = Surv.from_arrays(tst["event"].astype(bool), tst["time"].astype(float))

    alpha = 0.1
    mpath = Path(metrics)
    if mpath.exists():
        alpha = (
            json.loads(mpath.read_text()).get("tuning", {}).get("cox", {}).get("best_params", {}).get("alpha", alpha)
        )

    model = Pipeline([("prep", build_preprocessor(X_train)), ("model", CoxPHSurvivalAnalysis(alpha=alpha))])
    model.fit(X_train, y_train)

    km_t, km_s = kaplan_meier_estimator(y_train["event"], y_train["time"])
    cutoffs = np.quantile(model.predict(X_train), [0.25, 0.5, 0.75])
    c = concordance_index_ipcw(y_train, y_test, model.predict(X_test), tau=96)[0]
    return ServingModel(model, X_train, km_t, km_s, cutoffs, float(c), float(alpha))
