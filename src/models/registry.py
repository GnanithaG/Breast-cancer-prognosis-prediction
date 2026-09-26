"""Model definitions and validation-set tuning.

Every model is a scikit-learn ``Pipeline(preprocessor -> survival estimator)`` so
preprocessing is fitted on training rows only. A small hyper-parameter grid is
searched per model; the setting with the best Uno's C-index on the *validation*
split wins. The test split is never looked at here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw

log = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    label: str
    factory: Callable[..., Any]
    grid: dict[str, list] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    seeded: bool = False


MODELS: dict[str, ModelSpec] = {
    "cox": ModelSpec(
        name="cox",
        label="Cox PH (ridge)",
        factory=CoxPHSurvivalAnalysis,
        grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
    ),
    "rsf": ModelSpec(
        name="rsf",
        label="Random Survival Forest",
        factory=RandomSurvivalForest,
        grid={"min_samples_leaf": [5, 15, 30], "max_features": ["sqrt", 0.5]},
        fixed={"n_estimators": 300, "min_samples_split": 10, "n_jobs": -1},
        seeded=True,
    ),
    "gbsa": ModelSpec(
        name="gbsa",
        label="Gradient Boosted Cox",
        factory=GradientBoostingSurvivalAnalysis,
        grid={"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        fixed={"n_estimators": 200, "subsample": 0.8, "min_samples_leaf": 15},
        seeded=True,
    ),
}


def _param_grid(grid: dict[str, list]):
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def fit_and_tune(
    spec: ModelSpec,
    preprocessor,
    X_train,
    y_train,
    X_val,
    y_val,
    seed: int = 7,
    tau: float | None = None,
) -> tuple[Pipeline, dict]:
    """Try every grid point, keep the best on validation Uno's C, return it fitted on train."""
    best = (-np.inf, None, None)
    tried = []
    for params in _param_grid(spec.grid) if spec.grid else [{}]:
        kwargs = {**spec.fixed, **params}
        if spec.seeded:
            kwargs["random_state"] = seed
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", spec.factory(**kwargs))])
        pipe.fit(X_train, y_train)
        c_val = concordance_index_ipcw(y_train, y_val, pipe.predict(X_val), tau=tau)[0]
        tried.append({"params": params, "val_c_index_uno": round(float(c_val), 4)})
        log.info("  %s %s -> val C (Uno) = %.4f", spec.name, params, c_val)
        if c_val > best[0]:
            best = (c_val, pipe, params)

    c_val, pipe, params = best
    info = {"best_params": params, "val_c_index_uno": float(c_val), "grid": tried}
    return pipe, info
