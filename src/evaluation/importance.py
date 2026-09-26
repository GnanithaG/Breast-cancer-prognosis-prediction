from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_cindex(model, X: pd.DataFrame, y, n_repeats=10, seed=7):
    """Column-level permutation importance, scored with the model's C-index.

    Works on the *raw* columns (before one-hot encoding), so each clinical variable
    gets a single score.
    """
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    rows = sorted(zip(X.columns, r.importances_mean, r.importances_std), key=lambda t: t[1], reverse=True)
    return [(str(f), float(m), float(s)) for f, m, s in rows]
