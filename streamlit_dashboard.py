from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    tuning = pd.read_csv(OUTPUT_DIR / "tuning_summary.csv")
    roc = pd.read_csv(OUTPUT_DIR / "roc_curves.csv")
    confusion = pd.read_csv(OUTPUT_DIR / "confusion_matrices.csv")
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
    profile = json.loads((OUTPUT_DIR / "data_profile.json").read_text(encoding="utf-8"))
    return metrics, tuning, roc, confusion, importance, profile


@st.cache_resource
def load_model():
    return joblib.load(OUTPUT_DIR / "best_churn_model.joblib")


st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.caption("Business / Marketing / Supply Chain | Random Forest vs Logistic Regression vs SVM")

metrics, tuning, roc, confusion, importance, profile = load_tables()
best_model_name = profile["best_model"]
best_row = metrics.loc[metrics["Model"] == best_model_name].iloc[0]

kpi_cols = st.columns(5)
kpi_cols[0].metric("Selected model", best_model_name)
kpi_cols[1].metric("F1-score", f"{best_row['F1']:.3f}")
kpi_cols[2].metric("ROC-AUC", f"{best_row['ROC-AUC']:.3f}")
kpi_cols[3].metric("Recall", f"{best_row['Recall']:.3f}")
kpi_cols[4].metric("Sample rows", f"{profile['sample_rows']:,}")

tab_overview, tab_models, tab_inference = st.tabs(
    ["Overview", "Model Comparison", "Customer Inference"]
)

with tab_overview:
    st.subheader("Project Summary")
    st.write(
        "This dashboard summarizes a customer churn classification project. "
        "The goal is to identify customers who are likely to leave so a business team "
        "can prioritize retention actions."
    )

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.dataframe(metrics, width="stretch", hide_index=True)
    with col_right:
        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Features Used by {best_model_name}",
            color_discrete_sequence=["#2f6f73"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=430)
        st.plotly_chart(fig, width="stretch")

with tab_models:
    st.subheader("Model Performance")
    metric_choice = st.selectbox(
        "Metric to compare",
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        index=3,
    )
    fig = px.bar(
        metrics,
        x="Model",
        y=metric_choice,
        color="Model",
        text=metric_choice,
        title=f"{metric_choice} by Model",
        color_discrete_sequence=["#2f6f73", "#d95f43", "#4c78a8"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, width="stretch")

    st.subheader("ROC Curves")
    roc_fig = px.line(
        roc,
        x="FPR",
        y="TPR",
        color="Model",
        title="ROC Curve Comparison",
        color_discrete_sequence=["#2f6f73", "#d95f43", "#4c78a8"],
    )
    roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#666666"))
    st.plotly_chart(roc_fig, width="stretch")

    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Model", metrics["Model"].tolist())
    matrix = confusion[confusion["Model"] == selected_model].pivot(
        index="Actual", columns="Predicted", values="Count"
    )
    st.dataframe(matrix, width="stretch")

    with st.expander("Hyperparameter tuning summary"):
        st.dataframe(tuning, width="stretch", hide_index=True)

with tab_inference:
    st.subheader("Predict Churn for One Customer")
    model = load_model()

    inputs: dict[str, object] = {}
    left, right = st.columns(2)
    numeric_features = profile["numeric_features"]
    categorical_features = profile["categorical_features"]

    for i, feature in enumerate(numeric_features):
        info = profile["numeric_ranges"][feature]
        target_col = left if i % 2 == 0 else right
        inputs[feature] = target_col.number_input(
            feature,
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["median"]),
        )

    for i, feature in enumerate(categorical_features):
        choices = profile["categories"][feature]
        target_col = left if i % 2 == 0 else right
        inputs[feature] = target_col.selectbox(feature, choices)

    customer = pd.DataFrame([inputs], columns=profile["feature_columns"])
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(model.predict(customer)[0])

    st.divider()
    result_label = "Likely to churn" if prediction == 1 else "Likely to stay"
    st.metric("Prediction", result_label)
    st.progress(probability)
    st.write(f"Estimated churn probability: **{probability:.1%}**")

    st.dataframe(customer, width="stretch", hide_index=True)
