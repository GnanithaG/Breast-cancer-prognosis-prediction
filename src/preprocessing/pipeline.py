from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute + scale numerics; constant-impute + one-hot categoricals.

    Returns an *unfitted* transformer. Each model gets its own clone, and it is
    only ever fitted on training rows (inside the model pipeline).
    """
    num_cols, cat_cols = split_feature_types(X)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        verbose_feature_names_out=False,
    )
