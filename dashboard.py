from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "churn_prediction_model.pkl"
OUTPUT_DIR = PROJECT_DIR / "outputs"

PAGE_TITLE = "Customer Churn AI Studio"
ACCENT = "#2f80ed"
TEAL = "#00a88f"
CORAL = "#ff6b5f"
INK = "#172033"
MUTED = "#697386"
SURFACE = "#ffffff"
SOFT = "#f5f8fc"
MODEL_COLORS = {
    "Logistic Regression": "#2f80ed",
    "KNN": "#00a88f",
    "Random Forest": "#ff6b5f",
}


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


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --ink: #172033;
            --muted: #697386;
            --surface: #ffffff;
            --soft: #f5f8fc;
            --line: #dce6f2;
            --blue: #2f80ed;
            --teal: #00a88f;
            --coral: #ff6b5f;
        }

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(47, 128, 237, 0.12), transparent 32rem),
                linear-gradient(180deg, #f7faff 0%, #eef4f9 45%, #f8fafc 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102033 0%, #162b42 58%, #213a54 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        section[data-testid="stSidebar"] * {
            color: #edf6ff;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            color: #d5e7f7 !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 0.72rem 0.85rem;
            margin: 0.34rem 0;
        }

        .brand-panel {
            padding: 1.15rem 1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(47, 128, 237, 0.3), rgba(0, 168, 143, 0.18));
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.16);
            margin-bottom: 1.1rem;
        }

        .brand-kicker, .eyebrow {
            color: #61d6c5;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .brand-title {
            font-size: 1.52rem;
            font-weight: 800;
            line-height: 1.12;
            margin: 0.32rem 0;
        }

        .brand-copy, .sidebar-footer {
            color: #c7d6e6;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .sidebar-footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.14);
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.1rem;
            border-radius: 24px;
            color: white;
            background:
                linear-gradient(135deg, rgba(16, 32, 51, 0.97), rgba(24, 70, 107, 0.92)),
                linear-gradient(45deg, #2f80ed, #00a88f);
            box-shadow: 0 26px 70px rgba(23, 32, 51, 0.22);
            margin-bottom: 1.15rem;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 420px;
            height: 420px;
            right: -160px;
            top: -180px;
            background: conic-gradient(from 180deg, rgba(47,128,237,.35), rgba(0,168,143,.35), rgba(255,107,95,.3), rgba(47,128,237,.35));
            filter: blur(8px);
            opacity: 0.78;
        }

        .hero > * {
            position: relative;
            z-index: 1;
        }

        .hero h1 {
            font-size: 2.45rem;
            line-height: 1.06;
            font-weight: 800;
            margin: 0.45rem 0 0.75rem;
            letter-spacing: 0;
        }

        .hero p {
            max-width: 760px;
            color: #dcecff;
            font-size: 1.02rem;
            line-height: 1.7;
            margin: 0;
        }

        .metric-card, .content-card, .insight-card, .result-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(205, 218, 232, 0.9);
            border-radius: 18px;
            box-shadow: 0 18px 46px rgba(23, 32, 51, 0.08);
        }

        .metric-card {
            padding: 1rem 1.05rem;
            min-height: 122px;
        }

        .metric-label {
            color: var(--muted);
            font-weight: 700;
            font-size: 0.76rem;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--ink);
            font-weight: 800;
            font-size: 1.75rem;
            line-height: 1.14;
            margin-top: 0.25rem;
        }

        .metric-value.long-value {
            font-size: 1.25rem;
            line-height: 1.22;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.2rem;
        }

        .content-card, .insight-card, .result-card {
            padding: 1.15rem 1.2rem;
        }

        .section-title {
            margin: 1.8rem 0 0.75rem;
        }

        .section-title h2 {
            color: var(--ink);
            font-size: 1.42rem;
            margin: 0.12rem 0;
            font-weight: 800;
        }

        .section-title p {
            color: var(--muted);
            margin: 0;
            line-height: 1.55;
        }

        .highlight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .highlight-card {
            padding: 1.05rem;
            border-radius: 18px;
            background: var(--surface);
            border: 1px solid var(--line);
            box-shadow: 0 12px 32px rgba(23, 32, 51, 0.07);
        }

        .highlight-card h3 {
            margin: 0 0 0.4rem;
            font-size: 1rem;
            color: var(--ink);
        }

        .highlight-card p {
            color: var(--muted);
            line-height: 1.55;
            margin: 0;
            font-size: 0.92rem;
        }

        .result-card {
            border-left: 7px solid var(--blue);
        }

        .risk-high { border-left-color: var(--coral); }
        .risk-medium { border-left-color: #f4b740; }
        .risk-low { border-left-color: var(--teal); }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(220, 230, 242, 0.95);
            border-radius: 16px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 12px 30px rgba(23, 32, 51, 0.07);
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(220, 230, 242, 0.95);
        }

        div.stButton > button, div.stDownloadButton > button {
            border-radius: 12px;
            border: 0;
            background: linear-gradient(135deg, #2f80ed, #00a88f);
            color: #ffffff;
            font-weight: 800;
            padding: 0.7rem 1rem;
            box-shadow: 0 12px 28px rgba(47, 128, 237, 0.26);
        }

        div[data-baseweb="select"] > div, input, textarea {
            border-radius: 12px !important;
        }

        @media (max-width: 900px) {
            .hero { padding: 1.45rem; }
            .hero h1 { font-size: 1.85rem; }
            .highlight-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    value_class = "metric-value long-value" if len(value) > 13 else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, body: str = "") -> None:
    body_html = f"<p>{body}</p>" if body else ""
    st.markdown(
        f"""
        <div class="section-title">
            <div class="eyebrow">{kicker}</div>
            <h2>{title}</h2>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(profile: dict, best_model_name: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">AI Retention Analytics</div>
            <h1>Customer Churn Prediction Command Center</h1>
            <p>
                A polished machine-learning dashboard for explaining churn risk, comparing
                model performance, and turning raw customer profiles into actionable retention
                signals. The current production candidate is <b>{best_model_name}</b>, trained
                and evaluated from the project dataset artifacts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK),
        title=dict(font=dict(size=18, color=INK), x=0.02),
        margin=dict(l=20, r=20, t=60, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#e7eef6", zerolinecolor="#dce6f2")
    fig.update_yaxes(gridcolor="#e7eef6", zerolinecolor="#dce6f2")
    return fig


def risk_bucket(probability: float) -> tuple[str, str, str]:
    if probability >= 0.7:
        return "High risk", "risk-high", "Immediate retention intervention recommended."
    if probability >= 0.4:
        return "Moderate risk", "risk-medium", "Monitor engagement and consider a targeted offer."
    return "Low risk", "risk-low", "Customer profile currently looks stable."


def sidebar(page_names: list[str]) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-panel">
                <div class="brand-kicker">CS485 AI Project</div>
                <div class="brand-title">Churn AI Studio</div>
                <div class="brand-copy">
                    Premium Streamlit dashboard for customer churn analytics, model comparison,
                    and retention-focused predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected = st.radio("Navigation", page_names, label_visibility="collapsed")
        st.markdown(
            """
            <div class="sidebar-footer">
                Built for the Artificial Intelligence lab project.<br>
                Dataset, model artifacts, and dashboard outputs load from local project folders.
            </div>
            """,
            unsafe_allow_html=True,
        )
    return selected


def render_overview(metrics: pd.DataFrame, importance: pd.DataFrame, profile: dict, best_model_name: str) -> None:
    best_row = metrics.loc[metrics["Model"] == best_model_name].iloc[0]
    render_hero(profile, best_model_name)

    cols = st.columns(5)
    with cols[0]:
        metric_card("Selected model", best_model_name, "Best final F1-score")
    with cols[1]:
        metric_card("F1-score", f"{best_row['F1']:.3f}", "Balances precision and recall")
    with cols[2]:
        metric_card("Recall", f"{best_row['Recall']:.3f}", "Captures true churners")
    with cols[3]:
        metric_card("ROC-AUC", f"{best_row['ROC-AUC']:.3f}", "Ranking quality")
    with cols[4]:
        metric_card("Evaluation rows", f"{profile['test_rows_used']:,}", "Held-out test sample")

    section_header(
        "Executive Summary",
        "A retention dashboard built for decisions",
        "The product view connects academic model evaluation with business-facing churn interpretation.",
    )
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown(
            """
            <div class="content-card">
                <b>Business objective.</b> Predict which customers are likely to leave so support,
                marketing, and customer success teams can prioritize retention actions before
                revenue is lost.
                <br><br>
                <b>Modeling approach.</b> Logistic Regression, KNN, and Random Forest are compared
                using accuracy, precision, recall, F1-score, and ROC-AUC. The final choice favors
                churn detection quality over accuracy alone.
                <br><br>
                <b>Deployment behavior.</b> The dashboard loads a complete preprocessing-plus-model
                pipeline, allowing raw customer profiles and uploaded CSV files to be scored directly.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        fig = px.bar(
            importance.head(12),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Churn Drivers for {best_model_name}",
            color="Importance",
            color_continuous_scale=["#d9f0f7", ACCENT],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(style_plotly(fig, 405), use_container_width=True)

    st.markdown(
        """
        <div class="highlight-grid">
            <div class="highlight-card">
                <h3>Retention-ready predictions</h3>
                <p>Single-customer scoring translates model output into risk levels and action-oriented interpretation.</p>
            </div>
            <div class="highlight-card">
                <h3>Transparent evaluation</h3>
                <p>Model comparison views expose final scores, ROC curves, tuning results, and confusion matrices.</p>
            </div>
            <div class="highlight-card">
                <h3>Operational batch scoring</h3>
                <p>CSV upload support validates required fields, scores every row, and returns a downloadable prediction file.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_explorer(profile: dict) -> None:
    train_preview, test_preview = load_dataset_preview()
    dataset_choice = st.selectbox("Dataset sample", ["Training", "Testing"])
    data = train_preview.copy() if dataset_choice == "Training" else test_preview.copy()

    section_header(
        "Data Exploration",
        "Inspect customer behavior and churn patterns",
        "Interactive filters make the dataset easier to reason about before looking at model results.",
    )

    filter_cols = st.columns(4)
    with filter_cols[0]:
        genders = st.multiselect("Gender", sorted(data["Gender"].dropna().unique()), default=sorted(data["Gender"].dropna().unique()))
    with filter_cols[1]:
        subscriptions = st.multiselect(
            "Subscription",
            sorted(data["Subscription Type"].dropna().unique()),
            default=sorted(data["Subscription Type"].dropna().unique()),
        )
    with filter_cols[2]:
        contracts = st.multiselect(
            "Contract",
            sorted(data["Contract Length"].dropna().unique()),
            default=sorted(data["Contract Length"].dropna().unique()),
        )
    with filter_cols[3]:
        max_support = int(data["Support Calls"].max())
        support_range = st.slider("Support calls", 0, max_support, (0, max_support))

    filtered = data[
        data["Gender"].isin(genders)
        & data["Subscription Type"].isin(subscriptions)
        & data["Contract Length"].isin(contracts)
        & data["Support Calls"].between(support_range[0], support_range[1])
    ].copy()

    cols = st.columns(4)
    with cols[0]:
        metric_card("Visible rows", f"{len(filtered):,}", f"{dataset_choice.lower()} preview")
    with cols[1]:
        churn_rate = filtered["Churn"].mean() if len(filtered) else 0
        metric_card("Churn rate", f"{churn_rate:.1%}", "Within filtered sample")
    with cols[2]:
        metric_card("Avg spend", f"${filtered['Total Spend'].mean():,.0f}" if len(filtered) else "$0", "Customer value signal")
    with cols[3]:
        metric_card("Avg support", f"{filtered['Support Calls'].mean():.1f}" if len(filtered) else "0.0", "Service friction")

    left, right = st.columns(2)
    with left:
        age_fig = px.histogram(
            filtered,
            x="Age",
            color="Churn",
            nbins=24,
            barmode="overlay",
            title="Age Distribution by Churn",
            color_discrete_map={0: TEAL, 1: CORAL},
        )
        st.plotly_chart(style_plotly(age_fig), use_container_width=True)

    with right:
        spend_fig = px.box(
            filtered,
            x="Contract Length",
            y="Total Spend",
            color="Churn",
            title="Spend Distribution by Contract",
            color_discrete_map={0: TEAL, 1: CORAL},
        )
        st.plotly_chart(style_plotly(spend_fig), use_container_width=True)

    lower, upper = st.columns([0.95, 1.05])
    with lower:
        churn_by_contract = (
            filtered.groupby("Contract Length", as_index=False)["Churn"].mean().sort_values("Churn", ascending=False)
            if len(filtered)
            else pd.DataFrame(columns=["Contract Length", "Churn"])
        )
        contract_fig = px.bar(
            churn_by_contract,
            x="Contract Length",
            y="Churn",
            title="Churn Rate by Contract Length",
            color="Contract Length",
            color_discrete_sequence=[ACCENT, TEAL, CORAL],
        )
        contract_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_plotly(contract_fig, 380), use_container_width=True)
    with upper:
        st.markdown(
            f"""
            <div class="insight-card">
                <b>Dataset files</b><br>
                Training: <code>{profile['training_file']}</code><br>
                Testing: <code>{profile['testing_file']}</code><br><br>
                <b>Available rows</b><br>
                {profile['train_rows_available']:,} training rows and {profile['test_rows_available']:,} testing rows.
                The dashboard previews 5,000 rows for fast interactive exploration while keeping the full project
                artifacts available for model evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(filtered.head(25), use_container_width=True, hide_index=True)


def render_model_performance(
    metrics: pd.DataFrame,
    baseline: pd.DataFrame,
    tuning: pd.DataFrame,
    roc: pd.DataFrame,
    confusion: pd.DataFrame,
    best_model_name: str,
) -> None:
    section_header(
        "Model Performance",
        "Compare the final tuned classifiers",
        "The selected model is judged using held-out test results and business-aware metrics.",
    )

    metric_choice = st.selectbox("Metric to compare", ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], index=3)
    fig = px.bar(
        metrics,
        x="Model",
        y=metric_choice,
        color="Model",
        text=metric_choice,
        title=f"{metric_choice} by Final Model",
        color_discrete_map=MODEL_COLORS,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(style_plotly(fig), use_container_width=True)

    best_row = metrics.loc[metrics["Model"] == best_model_name].iloc[0]
    cols = st.columns(5)
    for col, metric_name in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        with col:
            metric_card(metric_name, f"{best_row[metric_name]:.3f}", best_model_name)

    left, right = st.columns([1.1, 0.9])
    with left:
        roc_fig = px.line(
            roc,
            x="FPR",
            y="TPR",
            color="Model",
            title="ROC Curve Comparison",
            color_discrete_map=MODEL_COLORS,
        )
        roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#697386"))
        st.plotly_chart(style_plotly(roc_fig), use_container_width=True)
    with right:
        selected_model = st.selectbox("Confusion matrix model", metrics["Model"].tolist(), index=metrics["Model"].tolist().index(best_model_name))
        matrix = confusion[confusion["Model"] == selected_model].pivot(index="Actual", columns="Predicted", values="Count")
        heatmap = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale=["#e9f6f5", TEAL],
            title=f"Confusion Matrix: {selected_model}",
            aspect="auto",
        )
        st.plotly_chart(style_plotly(heatmap, 420), use_container_width=True)

    st.markdown(
        f"""
        <div class="result-card">
            <b>Final model highlight: {best_model_name}</b><br>
            The selected model posts an F1-score of <b>{best_row['F1']:.3f}</b> and recall of
            <b>{best_row['Recall']:.3f}</b>. In churn prediction, recall is especially important
            because missed churners represent lost chances to intervene.
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_cols = st.columns(2)
    with table_cols[0]:
        st.subheader("Tuning Summary")
        st.dataframe(tuning, use_container_width=True, hide_index=True)
    with table_cols[1]:
        st.subheader("Untuned Baseline")
        st.dataframe(baseline, use_container_width=True, hide_index=True)


def collect_customer_inputs(profile: dict) -> pd.DataFrame:
    inputs: dict[str, object] = {}
    left, right = st.columns(2)

    for index, feature in enumerate(profile["numeric_features"]):
        info = profile["numeric_ranges"][feature]
        target_col = left if index % 2 == 0 else right
        min_value = float(info["min"])
        max_value = float(info["max"])
        median = float(info["median"])
        if min_value.is_integer() and max_value.is_integer():
            inputs[feature] = target_col.number_input(
                feature,
                min_value=int(min_value),
                max_value=int(max_value),
                value=int(round(median)),
                step=1,
            )
        else:
            inputs[feature] = target_col.number_input(feature, min_value=min_value, max_value=max_value, value=median)

    for index, feature in enumerate(profile["categorical_features"]):
        choices = profile["categories"][feature]
        target_col = left if index % 2 == 0 else right
        inputs[feature] = target_col.selectbox(feature, choices)

    return pd.DataFrame([inputs], columns=profile["feature_columns"])


def render_single_prediction(profile: dict, model) -> None:
    section_header(
        "Single Customer Prediction",
        "Score an individual customer profile",
        "The form uses the same raw features expected by the saved preprocessing and classifier pipeline.",
    )

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    customer = collect_customer_inputs(profile)
    st.markdown("</div>", unsafe_allow_html=True)

    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(model.predict(customer)[0])
    risk_label, risk_class, risk_note = risk_bucket(probability)
    result_label = "Likely to churn" if prediction == 1 else "Likely to stay"

    st.markdown(
        f"""
        <div class="result-card {risk_class}">
            <div class="metric-label">Prediction Result</div>
            <div class="metric-value">{result_label}</div>
            <div class="metric-note">{risk_label} - {probability:.1%} estimated churn probability. {risk_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(probability)

    summary_cols = st.columns(3)
    with summary_cols[0]:
        metric_card("Risk level", risk_label, "Probability bucket")
    with summary_cols[1]:
        metric_card("Churn probability", f"{probability:.1%}", "Model output")
    with summary_cols[2]:
        metric_card("Decision", result_label, "Classifier prediction")

    section_header("Input Summary", "Customer profile sent to the model")
    st.dataframe(customer, use_container_width=True, hide_index=True)


def validate_batch_columns(uploaded: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [column for column in feature_columns if column not in uploaded.columns]


def render_batch_prediction(profile: dict, model) -> None:
    section_header(
        "Batch Prediction",
        "Upload customer records and export scored churn risk",
        "The uploader accepts CSV files with the same feature columns used during training.",
    )

    uploaded_file = st.file_uploader("Upload a customer CSV file", type=["csv"])
    if uploaded_file is None:
        st.markdown(
            """
            <div class="content-card">
                Prepare a CSV with the required model features. Extra columns such as CustomerID are preserved
                in the preview and output, but the model scores only the validated feature set.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame({"Required feature": profile["feature_columns"]}), use_container_width=True, hide_index=True)
        return

    try:
        uploaded = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - Streamlit runtime guard
        st.error(f"Could not read the CSV file: {exc}")
        return

    missing = validate_batch_columns(uploaded, profile["feature_columns"])
    if missing:
        st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
        st.dataframe(pd.DataFrame({"Required feature": profile["feature_columns"]}), use_container_width=True, hide_index=True)
        return

    scoring_frame = uploaded[profile["feature_columns"]].copy()
    probabilities = model.predict_proba(scoring_frame)[:, 1]
    predictions = model.predict(scoring_frame)
    results = uploaded.copy()
    results["Churn Probability"] = probabilities
    results["Predicted Churn"] = predictions
    results["Risk Level"] = [risk_bucket(float(prob))[0] for prob in probabilities]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Rows scored", f"{len(results):,}", "Uploaded customers")
    with cols[1]:
        metric_card("Predicted churners", f"{int(predictions.sum()):,}", "Classifier output")
    with cols[2]:
        metric_card("Average risk", f"{probabilities.mean():.1%}", "Mean probability")
    with cols[3]:
        high_risk = int((probabilities >= 0.7).sum())
        metric_card("High risk", f"{high_risk:,}", "Probability >= 70%")

    dist_fig = px.histogram(
        results,
        x="Churn Probability",
        color="Risk Level",
        nbins=20,
        title="Predicted Churn Probability Distribution",
        color_discrete_map={"Low risk": TEAL, "Moderate risk": "#f4b740", "High risk": CORAL},
    )
    dist_fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style_plotly(dist_fig), use_container_width=True)

    st.dataframe(results.head(50), use_container_width=True, hide_index=True)
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored predictions",
        data=csv,
        file_name="customer_churn_batch_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="AI", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    try:
        metrics, baseline, tuning, roc, confusion, importance, profile = load_artifacts()
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Open `customer_churn_lab_project.ipynb`, run all cells, then refresh this dashboard.")
        st.stop()

    best_model_name = profile["best_model"]
    pages = ["Overview", "Data Exploration", "Model Performance", "Single Prediction", "Batch Prediction"]
    selected = sidebar(pages)

    if selected == "Overview":
        render_overview(metrics, importance, profile, best_model_name)
    elif selected == "Data Exploration":
        render_data_explorer(profile)
    elif selected == "Model Performance":
        render_model_performance(metrics, baseline, tuning, roc, confusion, best_model_name)
    elif selected == "Single Prediction":
        render_single_prediction(profile, model)
    else:
        render_batch_prediction(profile, model)


if __name__ == "__main__":
    main()
