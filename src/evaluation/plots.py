import os
import numpy as np
import matplotlib.pyplot as plt


def _survival_probs_at_times(model, X, times):
    surv_funcs = model.predict_survival_function(X)
    result = []
    for fn in surv_funcs:
        probs = []
        for t in times:
            if t <= fn.x[0]:
                probs.append(float(fn.y[0]))
            elif t >= fn.x[-1]:
                probs.append(float(fn.y[-1]))
            else:
                probs.append(float(fn(t)))
        result.append(probs)
    return np.array(result)


def plot_calibration_horizons(model_name, model, X, y_surv, horizons, outdir, annotate=False):
    os.makedirs(outdir, exist_ok=True)

    surv_probs = _survival_probs_at_times(model, X, horizons)

    for i, h in enumerate(horizons):
        pred_risk = 1.0 - surv_probs[:, i]
        observed = ((y_surv["time"] <= h) & (y_surv["event"])).astype(int)

        order = np.argsort(pred_risk)
        pred_sorted = pred_risk[order]
        obs_sorted = observed[order]

        bins = np.array_split(np.arange(len(pred_sorted)), 10)

        x_vals = []
        y_vals = []
        for b in bins:
            if len(b) == 0:
                continue
            x_vals.append(float(np.mean(pred_sorted[b])))
            y_vals.append(float(np.mean(obs_sorted[b])))

        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        plt.scatter(x_vals, y_vals, label="Observed bins")
        plt.xlabel(f"Predicted event risk by {h} months")
        plt.ylabel(f"Observed event rate by {h} months")
        plt.title(f"Calibration at {h} months - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"calibration_{model_name}_{int(h)}m.png"), dpi=300)
        plt.close()


def plot_ipcw_reliability(model_name, model, X, y_surv, horizons, outdir, annotate=False):
    os.makedirs(outdir, exist_ok=True)

    surv_probs = _survival_probs_at_times(model, X, horizons)

    for i, h in enumerate(horizons):
        pred_risk = 1.0 - surv_probs[:, i]
        observed = ((y_surv["time"] <= h) & (y_surv["event"])).astype(int)

        order = np.argsort(pred_risk)
        pred_sorted = pred_risk[order]
        obs_sorted = observed[order]

        bins = np.array_split(np.arange(len(pred_sorted)), 10)

        x_vals = []
        y_vals = []
        for b in bins:
            if len(b) == 0:
                continue
            x_vals.append(float(np.mean(pred_sorted[b])))
            y_vals.append(float(np.mean(obs_sorted[b])))

        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        plt.plot(x_vals, y_vals, marker="o", label="Reliability")
        plt.xlabel(f"Predicted event risk by {h} months")
        plt.ylabel(f"Observed event rate by {h} months")
        plt.title(f"Reliability at {h} months - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"reliability_{model_name}_{int(h)}m.png"), dpi=300)
        plt.close()


def plot_km_by_risk(model_name, model, X, y_surv, n_groups, outdir, annotate=False):
    os.makedirs(outdir, exist_ok=True)

    from lifelines import KaplanMeierFitter

    times = y_surv["time"]
    events = y_surv["event"]

    ref_time = int(np.median(times))
    surv_probs = _survival_probs_at_times(model, X, [ref_time])
    risk = 1.0 - surv_probs[:, 0]

    quantiles = np.quantile(risk, np.linspace(0, 1, n_groups + 1))
    groups = np.digitize(risk, quantiles[1:-1], right=True)

    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    for g in range(n_groups):
        mask = groups == g
        if np.sum(mask) == 0:
            continue
        kmf.fit(times[mask], events[mask], label=f"Q{g+1} risk group")
        kmf.plot_survival_function(ci_show=True)

    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.title(f"Kaplan-Meier by predicted risk - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"km_by_risk_{model_name}.png"), dpi=300)
    plt.close()