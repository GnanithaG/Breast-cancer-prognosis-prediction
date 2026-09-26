import numpy as np
import pandas as pd
import pytest
from sksurv.util import Surv

from src.data.harmonize import harmonize_registry_codes, norm_grade, norm_receptor
from src.data.ingest import coerce_event, load_csv, to_snake
from src.data.splits import make_or_load_splits
from src.evaluation.metrics import evaluate_model, km_survival_at, valid_horizons
from src.models.registry import MODELS, fit_and_tune

# ---------- ingest ----------


@pytest.mark.parametrize("dtype", [object, "string", "str"])
def test_coerce_event_text_any_string_dtype(dtype):
    # Regression: on pandas 3 the old code mapped every 'Dead' to 0 (all censored).
    s = pd.Series(["Alive", "Dead", " dead ", "ALIVE"]).astype(dtype)
    assert coerce_event(s).tolist() == [0, 1, 1, 0]


def test_coerce_event_rejects_unknown():
    with pytest.raises(ValueError, match="Unrecognised"):
        coerce_event(pd.Series(["Alive", "Lost"]))


def test_to_snake_fixes_names():
    assert to_snake("T Stage ") == "t_stage"
    assert to_snake("Reginol Node Positive") == "regional_node_positive"
    assert to_snake("6th Stage") == "stage_6th"


def test_load_csv_drops_raw_outcome_columns(raw_csv):
    df = load_csv(raw_csv)
    # Regression: 'Survival Months' / 'Status' used to stay in X -> target leakage.
    assert {"time", "event"} <= set(df.columns)
    assert not {"Survival Months", "Status", "survival_months", "status"} & set(df.columns)
    assert df["event"].isin([0, 1]).all() and df["event"].sum() > 0


def test_load_csv_removes_duplicates(tmp_path, raw_csv):
    d = pd.read_csv(raw_csv)
    p = tmp_path / "dup.csv"
    pd.concat([d, d.iloc[:3]]).to_csv(p, index=False)
    assert len(load_csv(p)) == len(load_csv(raw_csv))


# ---------- harmonize ----------


def test_norm_grade():
    assert norm_grade(" anaplastic; Grade IV") == "4"
    assert norm_grade("Grade II") == "2"
    assert norm_grade(3) == "3"
    assert norm_grade(None) == "Unknown"


def test_norm_receptor():
    assert norm_receptor("Positive") == "Positive"
    assert norm_receptor("neg") == "Negative"
    assert norm_receptor("??") == "Unknown"


def test_harmonize_caps_positive_nodes():
    df = pd.DataFrame({"regional_node_positive": [5, 2], "regional_node_examined": [3, 4]})
    out = harmonize_registry_codes(df)
    assert out["regional_node_positive"].tolist() == [3, 2]


# ---------- splits ----------


def test_splits_are_disjoint_and_regenerate_on_new_data(tmp_path, clean_df):
    path = tmp_path / "s.json"
    s = make_or_load_splits(clean_df, path)
    tr, va, te = map(set, (s["train"], s["val"], s["test"]))
    assert not (tr & va or tr & te or va & te)
    assert len(tr | va | te) == len(clean_df)
    assert make_or_load_splits(clean_df, path)["test"] == s["test"]  # reused
    changed = clean_df.iloc[:-5]
    s2 = make_or_load_splits(changed, path)
    assert s2["fingerprint"] != s["fingerprint"]
    assert max(s2["train"] + s2["val"] + s2["test"]) < len(changed)


# ---------- metrics ----------


def test_valid_horizons_drops_out_of_range():
    y = Surv.from_arrays([True, False, True], [5.0, 50.0, 80.0])
    assert valid_horizons([12, 60, 120], y, y).tolist() == [12.0, 60.0]


def test_km_survival_is_monotone():
    y = Surv.from_arrays([True, True, False, True], [1.0, 2.0, 3.0, 4.0])
    s = km_survival_at(y, [0.5, 1, 2, 3, 4])
    assert s[0] == 1.0 and np.all(np.diff(s) <= 0)


def test_end_to_end_cox_beats_chance(tmp_path, clean_df):
    s = make_or_load_splits(clean_df, tmp_path / "splits.json")
    parts = {k: clean_df.loc[s[k]] for k in ("train", "val", "test")}
    X = {k: v.drop(columns=["time", "event"]) for k, v in parts.items()}
    y = {k: Surv.from_arrays(v.event.astype(bool), v.time.astype(float)) for k, v in parts.items()}
    model, info = fit_and_tune(MODELS["cox"], X["train"], y["train"], X["val"], y["val"], tau=60)
    m = evaluate_model(model, y["train"], X["test"], y["test"], [24, 60], n_boot=0)
    assert m["c_index_uno"] > 0.6
    assert m["ibs_skill_vs_km"] > 0
    assert 0 < m["by_horizon"]["60"]["brier"] < 0.25


# ---------- baseline, explanations, serving ----------


def test_stage_baseline_uses_only_stage(tmp_path, clean_df):
    s = make_or_load_splits(clean_df, tmp_path / "s.json")
    tr, va = clean_df.loc[s["train"]], clean_df.loc[s["val"]]
    X_tr, X_va = tr.drop(columns=["time", "event"]), va.drop(columns=["time", "event"])
    y_tr = Surv.from_arrays(tr.event.astype(bool), tr.time.astype(float))
    y_va = Surv.from_arrays(va.event.astype(bool), va.time.astype(float))
    model, _ = fit_and_tune(MODELS["stage"], X_tr, y_tr, X_va, y_va, tau=60)
    changed = X_va.copy()
    changed["age"] = 30  # a non-stage column must not affect the prediction
    np.testing.assert_allclose(model.predict(X_va), model.predict(changed))


def test_linear_shap_sums_to_risk_difference(serving):
    from src.evaluation.explain import linear_shap

    X = serving.X_train.iloc[:20]
    phi = linear_shap(serving.model, X, serving.X_train)
    expected = serving.model.predict(X) - serving.model.predict(serving.X_train).mean()
    np.testing.assert_allclose(phi.sum(axis=1), expected, atol=1e-8)


def test_hazard_ratio_reference_is_one(serving):
    from src.evaluation.explain import hazard_ratios

    hr = hazard_ratios(serving.model, serving.X_train)
    refs = hr[hr.level.str.contains("reference")]
    assert len(refs) > 0 and np.allclose(refs.hazard_ratio, 1.0)


def test_stage_from_tn_matches_data(clean_df):
    from src.models.serving import stage_from_tn

    derived = [stage_from_tn(t, n) for t, n in zip(clean_df.t_stage, clean_df.n_stage, strict=True)]
    assert derived == clean_df.stage_6th.tolist()


def test_serving_prediction_is_sane(serving):
    inputs = serving.X_train.iloc[0].to_dict()
    X = serving.complete(inputs)
    out = serving.predict(X, [12, 60, 96])
    assert np.all(np.diff(out["survival"]) <= 0) and 0 < out["survival"][-1] <= 1
    assert out["risk_group"] in {"Lowest", "Low-mid", "High-mid", "Highest"}
    assert serving.test_c_index > 0.65
