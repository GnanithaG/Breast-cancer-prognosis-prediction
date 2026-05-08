import numpy as np
from sksurv.metrics import concordance_index_censored, integrated_brier_score


def _risk_scores(model, X):
    try:
        return model.predict(X)
    except Exception:
        surv_funcs = model.predict_survival_function(X)
        return np.array([-fn.x[-1] for fn in surv_funcs])


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


def evaluate_models(
    models,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    horizons,
):
    metrics = {}

    for name, model in models.items():
        print(f"[metrics] Evaluating {name}")

        risk = _risk_scores(model, X_test)
        cindex = concordance_index_censored(
            y_test["event"],
            y_test["time"],
            risk,
        )[0]

        model_metrics = {
            "c_index_ipcw": float(cindex),
            "by_horizon": {},
        }

        try:
            surv_probs = _survival_probs_at_times(model, X_test, horizons)
            train_times = y_train["time"]

            valid_horizons = [
                h for h in horizons
                if h > float(np.min(train_times)) and h < float(np.max(train_times))
            ]

            if len(valid_horizons) >= 2:
                surv_probs_valid = _survival_probs_at_times(model, X_test, valid_horizons)
                ibs = integrated_brier_score(
                    y_train,
                    y_test,
                    surv_probs_valid,
                    np.array(valid_horizons),
                )
                model_metrics["ibs"] = float(ibs)
            else:
                model_metrics["ibs"] = None

            for i, h in enumerate(horizons):
                failure_prob = 1.0 - surv_probs[:, i]
                observed_event = ((y_test["time"] <= h) & (y_test["event"])).astype(int)
                brier = np.mean((observed_event - failure_prob) ** 2)

                model_metrics["by_horizon"][str(int(h))] = {
                    "cindex": float(cindex),
                    "brier": float(brier),
                    "mean_predicted_risk": float(np.mean(failure_prob)),
                }

        except Exception as e:
            print(f"[metrics] Horizon metrics failed for {name}: {e}")
            for h in horizons:
                model_metrics["by_horizon"][str(int(h))] = {
                    "cindex": float(cindex),
                    "brier": None,
                    "mean_predicted_risk": None,
                }

        metrics[name] = model_metrics

    return metrics