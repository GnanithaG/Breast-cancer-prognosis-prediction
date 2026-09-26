"""Figures for the model report. All observed rates are Kaplan-Meier based, so censored
patients are handled correctly (a patient censored at 20 months is *not* counted as a
survivor at 60 months)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sksurv.metrics import brier_score, cumulative_dynamic_auc  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402

from src.evaluation.metrics import km_survival_at, survival_at  # noqa: E402

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
GREY = "#6b7280"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def _km_at(time, event, t):
    if event.sum() == 0:
        return 1.0
    x, s = kaplan_meier_estimator(event, time)
    i = np.searchsorted(x, t, side="right") - 1
    return float(s[i]) if i >= 0 else 1.0


def plot_calibration(model, label, X, y, horizons, path, n_bins=5):
    """Predicted vs KM-observed risk by predicted-risk quantile, one panel per horizon."""
    surv = survival_at(model, X, horizons)
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.4 * len(horizons), 3.4), squeeze=False)
    for ax, (i, h) in zip(axes[0], enumerate(horizons)):
        pred = 1 - surv[:, i]
        edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.digitize(pred, edges[1:-1], right=True), 0, n_bins - 1)
        px, oy = [], []
        for b in range(n_bins):
            m = bins == b
            if m.sum() < 5:
                continue
            px.append(pred[m].mean())
            oy.append(1 - _km_at(y["time"][m], y["event"][m], h))
        top = max(max(px + oy) * 1.1, 0.05)
        ax.plot([0, top], [0, top], ls="--", color=GREY, lw=1, label="Perfect")
        ax.plot(px, oy, "o-", color=PALETTE[0], label=label)
        ax.set(xlim=(0, top), ylim=(0, top), title=f"{int(h)} months", xlabel="Predicted risk")
        if i == 0:
            ax.set_ylabel("Observed risk (Kaplan-Meier)")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Calibration by risk quintile: {label}, test set", y=1.03)
    fig.savefig(path)
    plt.close(fig)


def plot_time_metrics(models, labels, y_train, X_test, y_test, path):
    """Time-dependent AUC and Brier score across follow-up, all models + KM baseline."""
    hi = np.percentile(y_test["time"], 95)
    times = np.linspace(6, hi, 40)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (name, model) in enumerate(models.items()):
        c = PALETTE[k % len(PALETTE)]
        auc, _ = cumulative_dynamic_auc(y_train, y_test, model.predict(X_test), times)
        _, bs = brier_score(y_train, y_test, survival_at(model, X_test, times), times)
        a1.plot(times, auc, color=c, label=labels[name])
        a2.plot(times, bs, color=c, label=labels[name])
    km = np.tile(km_survival_at(y_train, times), (len(y_test), 1))
    _, bs_km = brier_score(y_train, y_test, km, times)
    a2.plot(times, bs_km, color=GREY, ls="--", label="Kaplan-Meier (no covariates)")
    a1.axhline(0.5, color=GREY, ls="--", lw=1)
    a1.set(title="Time-dependent AUC (higher is better)", xlabel="Months", ylabel="AUC(t)")
    a2.set(title="Brier score (lower is better)", xlabel="Months", ylabel="Brier(t)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_km_by_risk(model, label, X, y, path, n_groups=4):
    risk = model.predict(X)
    edges = np.quantile(risk, np.linspace(0, 1, n_groups + 1))
    groups = np.clip(np.digitize(risk, edges[1:-1], right=True), 0, n_groups - 1)
    names = ["Lowest", "Low-mid", "High-mid", "Highest"] if n_groups == 4 else [f"Q{g + 1}" for g in range(n_groups)]
    colors = ["#059669", "#2563eb", "#d97706", "#dc2626"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for g in range(n_groups):
        m = groups == g
        t, s, ci = kaplan_meier_estimator(y["event"][m], y["time"][m], conf_type="log-log")
        c = colors[g % len(colors)]
        ax.step(t, s, where="post", color=c, label=f"{names[g]} risk (n={m.sum()}, deaths={y['event'][m].sum()})")
        ax.fill_between(t, ci[0], ci[1], step="post", color=c, alpha=0.12)
    ax.set(
        title=f"Observed survival by predicted risk quartile: {label}, test set",
        xlabel="Months since diagnosis",
        ylabel="Survival probability",
        ylim=(0, 1.02),
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def plot_importance(importance, label, path):
    """``importance``: list of (feature, mean, std) sorted descending."""
    feats = [f for f, _, _ in importance][::-1]
    means = [m for _, m, _ in importance][::-1]
    stds = [s for _, _, s in importance][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(feats) + 1))
    ax.barh(feats, means, xerr=stds, color=PALETTE[0], alpha=0.85)
    ax.axvline(0, color=GREY, lw=1)
    ax.set(title=f"Permutation importance: {label}, test set", xlabel="Drop in C-index when shuffled")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def save_all(models, labels, best, y_train, X_test, y_test, horizons, importance, figdir):
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    plot_time_metrics(models, labels, y_train, X_test, y_test, figdir / "time_dependent_metrics.png")
    plot_calibration(models[best], labels[best], X_test, y_test, horizons, figdir / "calibration.png")
    plot_km_by_risk(models[best], labels[best], X_test, y_test, figdir / "km_by_risk_group.png")
    plot_importance(importance, labels[best], figdir / "permutation_importance.png")


def plot_shap_summary(shap_df, X, label, path, max_features=12):
    """Beeswarm-style summary: one row per variable, one dot per patient."""
    order = shap_df.abs().mean().sort_values(ascending=True).index[-max_features:]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.2))
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for i, col in enumerate(order):
        vals = shap_df[col].to_numpy()
        jitter = rng.uniform(-0.28, 0.28, len(vals))
        if pd.api.types.is_numeric_dtype(X[col]):
            v = X[col].rank(pct=True).to_numpy()
            ax.scatter(vals, i + jitter, c=cmap(v), s=6, alpha=0.7, linewidths=0)
        else:
            ax.scatter(vals, i + jitter, color="#64748b", s=6, alpha=0.5, linewidths=0)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(range(len(order)), [f"{c}  ({shap_df[c].abs().mean():.2f})" for c in order])
    ax.set(
        xlabel="Contribution to log-hazard (SHAP), right = higher risk.  (n) = mean |SHAP|",
        title=f"What drives predicted risk: {label}",
    )
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("Numeric value (grey = categorical)", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_patient_waterfall(contrib, x_row, title, path, top=8):
    """Horizontal waterfall for one patient's SHAP values (log-hazard units)."""
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    head, rest = s.iloc[:top], s.iloc[top:]

    def fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)

    items = [(f"{k} = {fmt(x_row[k])}", v) for k, v in head.items()]
    if len(rest):
        items.append((f"{len(rest)} other variables", rest.sum()))
    n = len(items)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * n + 1.4))
    total = 0.0
    for idx, (_, v) in enumerate(items):  # largest first, drawn from the top down
        y = n - 1 - idx
        ax.barh(y, v, left=total, color="#dc2626" if v > 0 else "#2563eb", height=0.6)
        end = total + v
        ax.text(
            end + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8
        )
        total = end
    ax.set_yticks(range(n), [name for name, _ in items][::-1])
    ax.axvline(0, color=GREY, lw=1)
    ax.set(
        xlabel=f"Log-hazard vs average patient (total {total:+.2f}  =  {np.exp(total):.1f}x the average hazard)",
        title=title,
    )
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
