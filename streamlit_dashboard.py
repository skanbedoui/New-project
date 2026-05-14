from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "churn_prediction_model.pkl"
OUTPUT_DIR = PROJECT_DIR / "outputs"


@st.cache_data
def load_dataset_preview() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "customer_churn_dataset-training-master.csv"
    test_path = DATA_DIR / "customer_churn_dataset-testing-master.csv"
    return pd.read_csv(train_path, nrows=5000), pd.read_csv(test_path, nrows=5000)


@st.cache_data
def load_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    required_files = [
        OUTPUT_DIR / "model_metrics.csv",
        OUTPUT_DIR / "baseline_metrics.csv",
        OUTPUT_DIR / "tuning_summary.csv",
        OUTPUT_DIR / "roc_curves.csv",
        OUTPUT_DIR / "confusion_matrices.csv",
        OUTPUT_DIR / "feature_importance.csv",
        OUTPUT_DIR / "data_profile.json",
    ]
    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dashboard artifacts: "
            + ", ".join(missing)
            + ". Run the notebook from top to bottom first."
        )

    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_metrics.csv")
    tuning = pd.read_csv(OUTPUT_DIR / "tuning_summary.csv")
    roc = pd.read_csv(OUTPUT_DIR / "roc_curves.csv")
    confusion = pd.read_csv(OUTPUT_DIR / "confusion_matrices.csv")
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
    profile = json.loads((OUTPUT_DIR / "data_profile.json").read_text(encoding="utf-8"))
    return metrics, baseline, tuning, roc, confusion, importance, profile


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing model artifact models/churn_prediction_model.pkl. "
            "Run the notebook from top to bottom first."
        )
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.caption("CS280 / CS485 AI Lab | Logistic Regression vs KNN vs Random Forest")

try:
    metrics, baseline, tuning, roc, confusion, importance, profile = load_artifacts()
    model = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Open `customer_churn_lab_project.ipynb`, run all cells, then refresh this dashboard.")
    st.stop()

best_model_name = profile["best_model"]
best_row = metrics.loc[metrics["Model"] == best_model_name].iloc[0]

kpi_cols = st.columns(5)
kpi_cols[0].metric("Selected model", best_model_name)
kpi_cols[1].metric("F1-score", f"{best_row['F1']:.3f}")
kpi_cols[2].metric("Recall", f"{best_row['Recall']:.3f}")
kpi_cols[3].metric("ROC-AUC", f"{best_row['ROC-AUC']:.3f}")
kpi_cols[4].metric("Evaluation rows", f"{profile['test_rows_used']:,}")

tab_overview, tab_models, tab_data, tab_inference = st.tabs(
    ["Overview", "Model Comparison", "Dataset", "Customer Inference"]
)

with tab_overview:
    st.subheader("Project Summary")
    st.write(
        "This dashboard presents a supervised binary classification project for predicting customer churn. "
        "The final model is saved as a complete preprocessing and classifier pipeline, so raw customer "
        "profiles can be entered directly for inference."
    )

    left, right = st.columns([1.05, 1])
    with left:
        st.dataframe(metrics, width="stretch", hide_index=True)
    with right:
        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Features for {best_model_name}",
            color_discrete_sequence=["#4c78a8"],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=430)
        st.plotly_chart(fig, width="stretch")

with tab_models:
    st.subheader("Final Test Performance")
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
        color_discrete_sequence=["#4c78a8", "#f58518", "#54a24b"],
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
        color_discrete_sequence=["#4c78a8", "#f58518", "#54a24b"],
    )
    roc_fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash", color="#666666"),
    )
    st.plotly_chart(roc_fig, width="stretch")

    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Model", metrics["Model"].tolist())
    matrix = confusion[confusion["Model"] == selected_model].pivot(
        index="Actual", columns="Predicted", values="Count"
    )
    st.dataframe(matrix, width="stretch")

    with st.expander("Hyperparameter tuning summary"):
        st.dataframe(tuning, width="stretch", hide_index=True)

    with st.expander("Untuned baseline validation metrics"):
        st.dataframe(baseline, width="stretch", hide_index=True)

with tab_data:
    st.subheader("Dataset Files")
    train_preview, test_preview = load_dataset_preview()
    st.write(f"Training file: `{profile['training_file']}`")
    st.write(f"Testing file: `{profile['testing_file']}`")
    st.write(
        f"Available rows: {profile['train_rows_available']:,} training and "
        f"{profile['test_rows_available']:,} testing."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Training preview")
        st.dataframe(train_preview.head(20), width="stretch", hide_index=True)
    with col_b:
        st.caption("Testing preview")
        st.dataframe(test_preview.head(20), width="stretch", hide_index=True)

with tab_inference:
    st.subheader("Predict Churn for One Customer")

    inputs: dict[str, object] = {}
    left, right = st.columns(2)
    numeric_features = profile["numeric_features"]
    categorical_features = profile["categorical_features"]

    for index, feature in enumerate(numeric_features):
        info = profile["numeric_ranges"][feature]
        target_col = left if index % 2 == 0 else right
        if float(info["min"]).is_integer() and float(info["max"]).is_integer():
            inputs[feature] = target_col.number_input(
                feature,
                min_value=int(info["min"]),
                max_value=int(info["max"]),
                value=int(round(info["median"])),
                step=1,
            )
        else:
            inputs[feature] = target_col.number_input(
                feature,
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["median"]),
            )

    for index, feature in enumerate(categorical_features):
        choices = profile["categories"][feature]
        target_col = left if index % 2 == 0 else right
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
