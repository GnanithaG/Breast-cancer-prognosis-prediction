"""Interactive demo: predicted survival for a breast cancer patient.

streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.serving import GRADE_TO_DIFFERENTIATION, build_serving_model  # noqa: E402

st.set_page_config(page_title="Breast Cancer Survival Estimator", page_icon="🎗️", layout="wide")

LABELS = {
    "age": "Age (years)",
    "race": "Race",
    "marital_status": "Marital status",
    "t_stage": "T stage (tumour)",
    "n_stage": "N stage (lymph nodes)",
    "grade": "Grade (1 = well, 4 = undifferentiated)",
    "a_stage": "A stage",
    "tumor_size": "Tumour size (mm)",
    "estrogen_status": "Estrogen receptor",
    "progesterone_status": "Progesterone receptor",
    "regional_node_examined": "Lymph nodes examined",
    "regional_node_positive": "Lymph nodes positive",
}
SHORT = {
    **LABELS,
    "stage_6th": "AJCC stage",
    "grade": "Grade",
    "t_stage": "T stage",
    "n_stage": "N stage",
    "estrogen_status": "ER",
    "progesterone_status": "PR",
    "regional_node_positive": "Nodes positive",
    "regional_node_examined": "Nodes examined",
    "tumor_size": "Tumour size (mm)",
    "age": "Age",
}
HORIZONS = [12, 36, 60, 96]


@st.cache_resource(show_spinner="Training the model...")
def load():
    return build_serving_model()


sm = load()
schema = sm.schema

st.title("Breast cancer survival estimator")
st.caption(
    "Cox proportional hazards model trained on 2,816 patients from the SEER registry. "
    f"Test-set C-index {sm.test_c_index:.2f}. **Educational demo, not medical advice.**"
)

# ---------------- inputs ----------------
with st.sidebar:
    st.header("Patient")
    inputs = {}
    for col, label in LABELS.items():
        spec = schema[col]
        if spec["type"] == "numeric":
            inputs[col] = st.slider(label, spec["min"], spec["max"], spec["median"])
        else:
            levels = sorted(spec["levels"], key=str)
            default = levels.index(spec["levels"][0])  # most common level
            inputs[col] = st.selectbox(label, levels, index=default)
    if inputs["regional_node_positive"] > inputs["regional_node_examined"]:
        st.warning("Positive nodes can't exceed nodes examined; using the examined count.")
        inputs["regional_node_positive"] = inputs["regional_node_examined"]

X = sm.complete(inputs)
times = np.linspace(1, 107, 200)
pred = sm.predict(X, times)
pop = sm.population_survival(times)

# ---------------- headline ----------------
st.markdown(
    f"**Derived:** AJCC stage **{X['stage_6th'].iloc[0]}** · {GRADE_TO_DIFFERENTIATION[str(X['grade'].iloc[0])]} · "
    f"risk group **{pred['risk_group']}** (quartile {pred['risk_quartile']} of 4)"
)
cols = st.columns(len(HORIZONS))
for c, h in zip(cols, HORIZONS, strict=True):
    s = float(np.interp(h, times, pred["survival"]))
    p = float(sm.population_survival([h])[0])
    d = round((s - p) * 100)
    c.metric(
        f"{h // 12}-year survival" if h % 12 == 0 else f"{h}-month",
        f"{s:.0%}",
        f"{d:+d} pts vs average" if d else "same as average",
        delta_color="normal" if d else "off",
    )

left, right = st.columns([1.2, 1])

# ---------------- survival curve ----------------
with left:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(times, pred["survival"], color="#dc2626", lw=2.2, label="This patient")
    ax.step(times, pop, where="post", color="#6b7280", ls="--", label="All patients (Kaplan-Meier)")
    ax.set(xlabel="Months since diagnosis", ylabel="Probability of being alive", ylim=(0, 1.02), xlim=(0, 107))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left")
    st.pyplot(fig)
    plt.close(fig)

# ---------------- explanation ----------------
with right:
    st.subheader("What drives this estimate")
    shap = pred["shap"]
    top = shap.reindex(shap.abs().sort_values(ascending=False).index).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    ax.barh(
        [f"{SHORT.get(k, k)} = {X[k].iloc[0]}" for k in top.index],
        top.values,
        color=["#dc2626" if v > 0 else "#2563eb" for v in top.values],
    )
    ax.axvline(0, color="#6b7280", lw=1)
    ax.set_xlabel("← lowers risk        raises risk →", fontsize=11)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)
    rel = float(np.exp(pred["shap"].sum()))
    st.caption(
        f"Overall this patient's hazard is **{rel:.1f}x** that of an average patient in the data. "
        "Bars are exact SHAP values for the Cox model (log-hazard units)."
    )

with st.expander("About this model and its limits"):
    st.markdown(
        """
- Data: SEER breast cancer extract, women diagnosed 2006-2010, **all node-positive (N1-N3) and aged
  30-69**. Predictions outside those groups are not supported.
- Outcome is death from any cause over up to ~9 years of follow-up.
- Treatment, HER2 status and comorbidities are not in the data, so the model can't account for them.
- The AJCC stage group is derived automatically from T and N stage.
- Source code and full evaluation: [GitHub](https://github.com/GnanithaG/Breast-cancer-prognosis-prediction).
"""
    )
