"""Censoring-aware evaluation of survival models.

Metrics (all computed on the held-out TEST split):
  * Harrell's C-index            - classic ranking accuracy
  * Uno's C-index (IPCW)         - ranking accuracy corrected for censoring, truncated at tau
  * time-dependent AUC(t)        - can the model separate who dies by t from who survives past t
  * Brier score(t) (IPCW)        - calibration + discrimination of the predicted probability at t
  * Integrated Brier score (IBS) - Brier averaged over follow-up
  * IBS skill vs Kaplan-Meier    - 1 - IBS_model / IBS_KM; > 0 means better than a no-covariate model
95% CIs for the headline numbers come from a non-parametric bootstrap of the test set.
"""

from __future__ import annotations

import logging

import numpy as np
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator

log = logging.getLogger(__name__)


def survival_at(model, X, times) -> np.ndarray:
    """Predicted S(t | x) for each row of X at each time in ``times`` -> (n, len(times))."""
    times = np.asarray(times, dtype=float)
    fns = model.predict_survival_function(X)
    out = np.empty((len(fns), len(times)))
    for i, fn in enumerate(fns):
        lo, hi = fn.domain
        out[i] = fn(np.clip(times, lo, hi))
    return out


def km_survival_at(y_train, times) -> np.ndarray:
    """Kaplan-Meier survival curve from the training set evaluated at ``times``."""
    t, s = kaplan_meier_estimator(y_train["event"], y_train["time"])
    idx = np.searchsorted(t, times, side="right") - 1
    return np.where(idx >= 0, s[np.clip(idx, 0, None)], 1.0)


def valid_horizons(horizons, y_train, y_test) -> np.ndarray:
    """Keep horizons strictly inside the follow-up range of both train and test."""
    upper = min(y_train["time"].max(), y_test["time"].max())
    lower = max(y_train["time"].min(), y_test["time"].min())
    keep = np.array([h for h in horizons if lower < h < upper], dtype=float)
    dropped = sorted(set(map(float, horizons)) - set(keep))
    if dropped:
        log.warning("Dropping horizons outside follow-up range (%.0f-%.0f months): %s", lower, upper, dropped)
    if len(keep) == 0:
        raise ValueError("No evaluation horizon falls inside the follow-up range.")
    return keep


def ibs_grid(y_test, tau: float, n: int = 50) -> np.ndarray:
    lo = np.percentile(y_test["time"], 5)
    return np.linspace(lo, tau, n)


def _core(y_train, y_test, risk, surv_grid, grid, tau):
    c_uno = concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0]
    ibs = integrated_brier_score(y_train, y_test, surv_grid, grid)
    return c_uno, ibs


def evaluate_model(model, y_train, X_test, y_test, horizons, n_boot=200, seed=7) -> dict:
    horizons = np.asarray(horizons, dtype=float)
    tau = float(horizons.max())
    grid = ibs_grid(y_test, tau)

    risk = model.predict(X_test)
    surv_h = survival_at(model, X_test, horizons)
    surv_grid = survival_at(model, X_test, grid)

    c_harrell = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
    c_uno, ibs = _core(y_train, y_test, risk, surv_grid, grid, tau)
    auc_t, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, horizons)
    _, brier_t = brier_score(y_train, y_test, surv_h, horizons)

    km_grid = np.tile(km_survival_at(y_train, grid), (len(y_test), 1))
    ibs_km = integrated_brier_score(y_train, y_test, km_grid, grid)

    # Bootstrap the test set for CIs on the headline metrics.
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y_test)
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yb = y_test[b]
        if yb["event"].sum() < 2 or yb["time"].max() <= tau:
            continue
        try:
            boots.append(_core(y_train, yb, risk[b], surv_grid[b], grid, tau))
        except ValueError:
            continue
    boots = np.array(boots)

    def ci(col):
        if len(boots) < 20:
            return [None, None]
        return [float(v) for v in np.percentile(boots[:, col], [2.5, 97.5])]

    return {
        "c_index_harrell": float(c_harrell),
        "c_index_uno": float(c_uno),
        "c_index_uno_ci95": ci(0),
        "tau_months": tau,
        "mean_auc": float(mean_auc),
        "ibs": float(ibs),
        "ibs_ci95": ci(1),
        "ibs_kaplan_meier": float(ibs_km),
        "ibs_skill_vs_km": float(1 - ibs / ibs_km),
        "n_bootstrap": int(len(boots)),
        "by_horizon": {
            str(int(h)): {
                "auc": float(auc_t[i]),
                "brier": float(brier_t[i]),
                "mean_predicted_risk": float(np.mean(1 - surv_h[:, i])),
            }
            for i, h in enumerate(horizons)
        },
    }


def evaluate_models(models: dict, y_train, X_test, y_test, horizons, n_boot=200, seed=7) -> dict:
    out = {}
    for name, model in models.items():
        log.info("Evaluating %s on the test set", name)
        out[name] = evaluate_model(model, y_train, X_test, y_test, horizons, n_boot=n_boot, seed=seed)
    return out


def paired_c_index_difference(y_train, y_test, risk_a, risk_b, tau, n_boot=500, seed=7) -> dict:
    """Uno's C of model A minus model B on the same test patients, with a paired bootstrap CI.

    Resampling the *same* patients for both models removes most of the noise that makes
    the separate confidence intervals overlap, so this is the right test for "is A better than B?".
    """
    c_a = concordance_index_ipcw(y_train, y_test, risk_a, tau=tau)[0]
    c_b = concordance_index_ipcw(y_train, y_test, risk_b, tau=tau)[0]
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y_test)
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yb = y_test[b]
        if yb["event"].sum() < 2 or yb["time"].max() <= tau:
            continue
        try:
            diffs.append(
                concordance_index_ipcw(y_train, yb, risk_a[b], tau=tau)[0]
                - concordance_index_ipcw(y_train, yb, risk_b[b], tau=tau)[0]
            )
        except ValueError:
            continue
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) >= 20 else (np.nan, np.nan)
    # Two-sided bootstrap p-value: how often the difference falls on the other side of zero.
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))) if len(diffs) else float("nan")
    return {
        "c_a": float(c_a),
        "c_b": float(c_b),
        "difference": float(c_a - c_b),
        "ci95": [float(lo), float(hi)],
        "p_value": p,
        "n_bootstrap": int(len(diffs)),
    }
