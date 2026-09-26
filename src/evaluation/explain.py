"""Explanations for linear (Cox) survival pipelines.

For a Cox model the risk score is linear in the preprocessed features, so SHAP values
have an exact closed form (no sampling or approximation):

    phi_j(x) = beta_j * (z_j(x) - mean_background(z_j))

One-hot columns are summed back to their original clinical variable, so each patient
gets one contribution per variable. Contributions are in log-hazard units:
+0.69 means that variable roughly doubles this patient's hazard relative to an average patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _column_groups(prep) -> list[str]:
    """Original column name for every output column of the fitted ColumnTransformer."""
    groups = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if name == "num":
            groups += list(cols)
        else:
            onehot = trans.named_steps["onehot"]
            for col, cats in zip(cols, onehot.categories_, strict=True):
                groups += [col] * len(cats)
    return groups


def is_linear(pipe) -> bool:
    return hasattr(pipe.named_steps["model"], "coef_")


def linear_shap(pipe, X: pd.DataFrame, X_background: pd.DataFrame) -> pd.DataFrame:
    """Exact SHAP values per original column for a fitted Cox pipeline -> DataFrame like X."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    z = np.asarray(prep.transform(X), dtype=float)
    z_bg = np.asarray(prep.transform(X_background), dtype=float).mean(axis=0)
    contrib = (z - z_bg) * model.coef_
    groups = _column_groups(prep)
    df = pd.DataFrame(contrib, columns=groups, index=X.index)
    return df.T.groupby(level=0, sort=False).sum().T


def hazard_ratios(pipe, X_train: pd.DataFrame) -> pd.DataFrame:
    """Hazard ratios from a Cox pipeline.

    Categorical levels are compared with the most common level (the reference, HR = 1).
    Because exactly one level is active per patient, subtracting the reference coefficient
    gives the usual "vs reference" hazard ratio. Numeric variables are per 1 standard deviation.
    """
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = list(prep.get_feature_names_out())
    groups = _column_groups(prep)
    coef = dict(zip(names, model.coef_, strict=True))
    num = prep.named_transformers_["num"]
    scale = dict(zip(num.feature_names_in_, num.named_steps["scale"].scale_, strict=True))
    rows = []
    for grp in dict.fromkeys(groups):
        if grp in scale:
            b = coef[grp]
            rows.append({"variable": grp, "level": f"per SD (+{scale[grp]:.1f})", "hazard_ratio": float(np.exp(b))})
            continue
        ref = str(X_train[grp].mode().iloc[0])
        b_ref = coef[f"{grp}_{ref}"]
        for name, g in zip(names, groups, strict=True):
            if g != grp:
                continue
            level = name[len(grp) + 1 :]
            hr = float(np.exp(coef[name] - b_ref))
            rows.append(
                {"variable": grp, "level": level + (" (reference)" if level == ref else ""), "hazard_ratio": hr}
            )
    return pd.DataFrame(rows)
