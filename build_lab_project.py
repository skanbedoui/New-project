from __future__ import annotations

import base64
import json
import textwrap
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(
    r"C:\Users\Mega-PC\Downloads\customer_churn_dataset-training-master.csv"
    r"\customer_churn_dataset-training-master.csv"
)
NOTEBOOK_PATH = PROJECT_DIR / "customer_churn_lab_project.ipynb"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
RANDOM_STATE = 42
SAMPLE_SIZE = 25_000


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def image_output(path: Path) -> nbf.NotebookNode:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return new_output("display_data", data={"image/png": data}, metadata={})


def html_output(df: pd.DataFrame) -> nbf.NotebookNode:
    return new_output(
        "display_data",
        data={"text/html": df.to_html(index=False), "text/plain": df.to_string(index=False)},
        metadata={},
    )


def clean_feature_name(name: str) -> str:
    return (
        name.replace("num__", "")
        .replace("cat__", "")
        .replace("remainder__", "")
        .replace("_", " ")
    )


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.dropna().copy()
    df["Churn"] = df["Churn"].astype(int)
    df["CustomerID"] = df["CustomerID"].astype(int)

    feature_columns = [c for c in df.columns if c not in ["CustomerID", "Churn"]]
    numeric_features = df[feature_columns].select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    # A stratified sample keeps the notebook fast enough to rerun and keeps the
    # train/test comparison fair because every model sees the same records.
    if len(df) > SAMPLE_SIZE:
        _, sample_df = train_test_split(
            df,
            test_size=SAMPLE_SIZE,
            stratify=df["Churn"],
            random_state=RANDOM_STATE,
        )
    else:
        sample_df = df.copy()

    X = sample_df[feature_columns]
    y = sample_df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model_specs = {
        "Random Forest": (
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [150],
                "classifier__max_depth": [8, None],
            },
        ),
        "Logistic Regression": (
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                            max_iter=2000,
                            solver="liblinear",
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        ),
        "SVM": (
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "classifier",
                        CalibratedClassifierCV(
                            estimator=LinearSVC(
                                random_state=RANDOM_STATE,
                                class_weight="balanced",
                                dual="auto",
                                max_iter=5000,
                            ),
                            method="sigmoid",
                            cv=3,
                        ),
                    ),
                ]
            ),
            {"classifier__estimator__C": [0.1, 1.0]},
        ),
    }

    fitted_models = {}
    tuning_rows = []
    for model_name, (pipeline, param_grid) in model_specs.items():
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=3,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        fitted_models[model_name] = grid.best_estimator_
        tuning_rows.append(
            {
                "Model": model_name,
                "Best CV F1": round(float(grid.best_score_), 4),
                "Best parameters": json.dumps(grid.best_params_),
            }
        )

    metric_rows = []
    roc_rows = []
    confusion_rows = []
    for model_name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            }
        )
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_rows.extend(
            {
                "Model": model_name,
                "FPR": float(fpr_value),
                "TPR": float(tpr_value),
                "Threshold": float(threshold_value),
            }
            for fpr_value, tpr_value, threshold_value in zip(fpr, tpr, thresholds)
        )
        matrix = confusion_matrix(y_test, y_pred)
        for actual_idx, actual_label in enumerate(["No churn", "Churn"]):
            for pred_idx, pred_label in enumerate(["No churn", "Churn"]):
                confusion_rows.append(
                    {
                        "Model": model_name,
                        "Actual": actual_label,
                        "Predicted": pred_label,
                        "Count": int(matrix[actual_idx, pred_idx]),
                    }
                )

    tuning_df = pd.DataFrame(tuning_rows)
    metrics_df = pd.DataFrame(metric_rows).sort_values("F1", ascending=False)
    roc_df = pd.DataFrame(roc_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_model = fitted_models[best_model_name]

    feature_names = [
        clean_feature_name(name)
        for name in best_model.named_steps["preprocess"].get_feature_names_out()
    ]
    classifier = best_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance_values = np.abs(classifier.coef_).ravel()
    else:
        coefs = [
            calibrated.estimator.coef_.ravel()
            for calibrated in getattr(classifier, "calibrated_classifiers_", [])
            if hasattr(calibrated.estimator, "coef_")
        ]
        importance_values = np.mean(np.abs(coefs), axis=0) if coefs else np.zeros(len(feature_names))

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    prediction_examples = X_test.head(8).copy()
    prediction_examples.insert(0, "Actual Churn", y_test.head(8).to_numpy())
    prediction_examples["Predicted Churn"] = best_model.predict(X_test.head(8))
    prediction_examples["Churn Probability"] = best_model.predict_proba(X_test.head(8))[:, 1]

    metrics_rounded = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        metrics_rounded[col] = metrics_rounded[col].round(4)

    # Save reusable dashboard assets.
    metrics_rounded.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    tuning_df.to_csv(OUTPUT_DIR / "tuning_summary.csv", index=False)
    roc_df.to_csv(OUTPUT_DIR / "roc_curves.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    prediction_examples.to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)
    joblib.dump(best_model, OUTPUT_DIR / "best_churn_model.joblib")

    profile = {
        "data_path": str(DATA_PATH),
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df)),
        "sample_rows": int(len(sample_df)),
        "target_name": "Churn",
        "best_model": best_model_name,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }
            for col in numeric_features
        },
        "categories": {
            col: sorted([str(value) for value in df[col].dropna().unique().tolist()])
            for col in categorical_features
        },
    }
    (OUTPUT_DIR / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    # Figures for the notebook and dashboard.
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", hue="Churn", palette=["#2f6f73", "#d95f43"], legend=False)
    plt.title("Churn Target Distribution")
    plt.xlabel("Churn class")
    plt.ylabel("Number of customers")
    plt.xticks([0, 1], ["No churn", "Churn"])
    target_fig = FIG_DIR / "target_distribution.png"
    save_plot(target_fig)

    metric_long = metrics_rounded.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=metric_long, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison Across Evaluation Metrics")
    plt.legend(loc="lower right")
    metrics_fig = FIG_DIR / "model_comparison.png"
    save_plot(metrics_fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, model_name in zip(axes, fitted_models.keys()):
        sub = confusion_df[confusion_df["Model"] == model_name]
        matrix = sub.pivot(index="Actual", columns="Predicted", values="Count").loc[
            ["No churn", "Churn"], ["No churn", "Churn"]
        ]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=axis)
        axis.set_title(model_name)
    confusion_fig = FIG_DIR / "confusion_matrices.png"
    save_plot(confusion_fig)

    plt.figure(figsize=(7, 5))
    for model_name in fitted_models.keys():
        sub = roc_df[roc_df["Model"] == model_name]
        auc_value = metrics_rounded.loc[metrics_rounded["Model"] == model_name, "ROC-AUC"].iloc[0]
        plt.plot(sub["FPR"], sub["TPR"], label=f"{model_name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Random baseline")
    plt.title("ROC Curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    roc_fig = FIG_DIR / "roc_curves.png"
    save_plot(roc_fig)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", color="#2f6f73")
    plt.title(f"Top Features Used by {best_model_name}")
    importance_fig = FIG_DIR / "feature_importance.png"
    save_plot(importance_fig)

    notebook = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )

    cells = [
        new_markdown_cell(
            "# Customer Churn Prediction Lab Project\n\n"
            "**Domain:** Business / Marketing / Supply Chain  \n"
            "**Problem type:** supervised binary classification  \n"
            "**Goal:** predict whether a customer is likely to churn, then compare Random Forest, "
            "Logistic Regression, and SVM under the same preprocessing and test split.\n\n"
            "This notebook follows the lab guideline requirements: problem definition, dataset "
            "preparation, preprocessing, model tuning, evaluation, model selection, and inference "
            "on unseen examples."
        ),
        new_markdown_cell(
            "## 1. Problem Definition\n\n"
            "Customer churn means losing a customer. In a business or marketing setting, predicting "
            "churn helps the company decide which customers may need retention actions such as "
            "support follow-up, discounts, or contract renewal campaigns.\n\n"
            "The target variable is `Churn`: `1` means the customer churned and `0` means the "
            "customer stayed. Because missing churners can be expensive, we evaluate more than "
            "accuracy. Precision, recall, F1-score, and ROC-AUC are used together."
        ),
        new_code_cell(
            textwrap.dedent(
                f"""
                import json
                from pathlib import Path

                import joblib
                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import seaborn as sns
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.compose import ColumnTransformer
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.impute import SimpleImputer
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
                from sklearn.model_selection import GridSearchCV, train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.svm import LinearSVC

                RANDOM_STATE = {RANDOM_STATE}
                SAMPLE_SIZE = {SAMPLE_SIZE}
                DATA_PATH = Path(r"{DATA_PATH}")
                OUTPUT_DIR = Path("outputs")
                FIG_DIR = OUTPUT_DIR / "figures"
                OUTPUT_DIR.mkdir(exist_ok=True)
                FIG_DIR.mkdir(exist_ok=True)
                """
            ).strip()
        ),
        new_markdown_cell("## 2. Dataset Loading and First Look"),
        new_code_cell(
            textwrap.dedent(
                """
                df_raw = pd.read_csv(DATA_PATH)
                df = df_raw.dropna().copy()
                df["Churn"] = df["Churn"].astype(int)
                df["CustomerID"] = df["CustomerID"].astype(int)

                print("Raw shape:", df_raw.shape)
                print("Clean shape after dropping the single incomplete row:", df.shape)
                print("Columns:", list(df.columns))
                display(df.head())
                """
            ).strip(),
            outputs=[
                new_output(
                    "stream",
                    name="stdout",
                    text=(
                        f"Raw shape: {df_raw.shape}\n"
                        f"Clean shape after dropping the single incomplete row: {df.shape}\n"
                        f"Columns: {list(df.columns)}\n"
                    ),
                ),
                html_output(df.head()),
            ],
            execution_count=2,
        ),
        new_markdown_cell(
            "## 3. Exploratory Data Analysis\n\n"
            "The dataset contains customer demographics, usage behavior, support interactions, "
            "payment delay, subscription information, contract length, spending, and recency of "
            "interaction. These features are directly useful for churn prediction because they "
            "describe both customer value and customer friction."
        ),
        new_code_cell(
            textwrap.dedent(
                """
                target_summary = (
                    df["Churn"].value_counts(normalize=True)
                    .rename_axis("Churn")
                    .reset_index(name="Percentage")
                )
                target_summary["Percentage"] = (target_summary["Percentage"] * 100).round(2)
                display(target_summary)

                sns.countplot(data=df, x="Churn", hue="Churn", palette=["#2f6f73", "#d95f43"], legend=False)
                plt.title("Churn Target Distribution")
                plt.xlabel("Churn class")
                plt.ylabel("Number of customers")
                plt.xticks([0, 1], ["No churn", "Churn"])
                plt.show()
                """
            ).strip(),
            outputs=[
                html_output(
                    df["Churn"]
                    .value_counts(normalize=True)
                    .rename_axis("Churn")
                    .reset_index(name="Percentage")
                    .assign(Percentage=lambda d: (d["Percentage"] * 100).round(2))
                ),
                image_output(target_fig),
            ],
            execution_count=3,
        ),
        new_markdown_cell(
            "## 4. Preprocessing and Data Split\n\n"
            "The same preprocessing is used for all models to keep the comparison fair. Numeric "
            "features are median-imputed and scaled. Categorical features are mode-imputed and "
            "one-hot encoded. `CustomerID` is removed because it is an identifier, not a real "
            "behavioral predictor.\n\n"
            "The dataset is very large, so this notebook uses a stratified sample for modeling. "
            "This keeps the class balance similar to the full dataset and makes the SVM practical "
            "to rerun during presentation."
        ),
        new_code_cell(
            textwrap.dedent(
                """
                feature_columns = [c for c in df.columns if c not in ["CustomerID", "Churn"]]
                numeric_features = df[feature_columns].select_dtypes(include=np.number).columns.tolist()
                categorical_features = [c for c in feature_columns if c not in numeric_features]

                _, sample_df = train_test_split(
                    df,
                    test_size=SAMPLE_SIZE,
                    stratify=df["Churn"],
                    random_state=RANDOM_STATE,
                )

                X = sample_df[feature_columns]
                y = sample_df["Churn"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
                )

                print("Modeling sample size:", sample_df.shape)
                print("Training size:", X_train.shape)
                print("Testing size:", X_test.shape)
                print("Numeric features:", numeric_features)
                print("Categorical features:", categorical_features)
                """
            ).strip(),
            outputs=[
                new_output(
                    "stream",
                    name="stdout",
                    text=(
                        f"Modeling sample size: {sample_df.shape}\n"
                        f"Training size: {X_train.shape}\n"
                        f"Testing size: {X_test.shape}\n"
                        f"Numeric features: {numeric_features}\n"
                        f"Categorical features: {categorical_features}\n"
                    ),
                )
            ],
            execution_count=4,
        ),
        new_markdown_cell(
            "## 5. Models and Hyperparameter Tuning\n\n"
            "- **Random Forest:** an ensemble of decision trees. It captures non-linear patterns "
            "and is easy to explain with feature importance.\n"
            "- **Logistic Regression:** a simple linear classifier. It estimates churn probability "
            "with a weighted sum of the input features and is easy to explain.\n"
            "- **SVM:** a linear support vector machine. It searches for a separating boundary with "
            "a maximum margin. Probability calibration is added so ROC-AUC can be compared fairly.\n\n"
            "Grid search uses cross-validation and F1-score because churn prediction needs a balance "
            "between finding churners and avoiding too many false alarms."
        ),
        new_code_cell(
            textwrap.dedent(
                """
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ])
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ])
                preprocessor = ColumnTransformer([
                    ("num", numeric_pipeline, numeric_features),
                    ("cat", categorical_pipeline, categorical_features),
                ])

                model_specs = {
                    "Random Forest": (
                        Pipeline([
                            ("preprocess", preprocessor),
                            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)),
                        ]),
                        {"classifier__n_estimators": [150], "classifier__max_depth": [8, None]},
                    ),
                    "Logistic Regression": (
                        Pipeline([
                            ("preprocess", preprocessor),
                            ("classifier", LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced", max_iter=2000, solver="liblinear")),
                        ]),
                        {"classifier__C": [0.1, 1.0, 10.0]},
                    ),
                    "SVM": (
                        Pipeline([
                            ("preprocess", preprocessor),
                            ("classifier", CalibratedClassifierCV(estimator=LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", dual="auto", max_iter=5000), method="sigmoid", cv=3)),
                        ]),
                        {"classifier__estimator__C": [0.1, 1.0]},
                    ),
                }

                fitted_models = {}
                tuning_rows = []
                for model_name, (pipeline, param_grid) in model_specs.items():
                    grid = GridSearchCV(pipeline, param_grid, scoring="f1", cv=3, n_jobs=1, refit=True)
                    grid.fit(X_train, y_train)
                    fitted_models[model_name] = grid.best_estimator_
                    tuning_rows.append({
                        "Model": model_name,
                        "Best CV F1": round(float(grid.best_score_), 4),
                        "Best parameters": grid.best_params_,
                    })

                tuning_df = pd.DataFrame(tuning_rows)
                display(tuning_df)
                """
            ).strip(),
            outputs=[html_output(tuning_df)],
            execution_count=5,
        ),
        new_markdown_cell(
            "## 6. Evaluation and Scientific Comparison\n\n"
            "All models are evaluated on the same unseen test set. The final selection is based "
            "mainly on F1-score, while ROC-AUC shows how well each model separates churners from "
            "non-churners across thresholds."
        ),
        new_code_cell(
            textwrap.dedent(
                """
                metric_rows = []
                roc_rows = []
                confusion_rows = []

                for model_name, model in fitted_models.items():
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    metric_rows.append({
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred),
                        "Recall": recall_score(y_test, y_pred),
                        "F1": f1_score(y_test, y_pred),
                        "ROC-AUC": roc_auc_score(y_test, y_proba),
                    })

                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_rows.extend({"Model": model_name, "FPR": f, "TPR": t, "Threshold": th} for f, t, th in zip(fpr, tpr, thresholds))

                    matrix = confusion_matrix(y_test, y_pred)
                    for actual_idx, actual_label in enumerate(["No churn", "Churn"]):
                        for pred_idx, pred_label in enumerate(["No churn", "Churn"]):
                            confusion_rows.append({
                                "Model": model_name,
                                "Actual": actual_label,
                                "Predicted": pred_label,
                                "Count": int(matrix[actual_idx, pred_idx]),
                            })

                metrics_df = pd.DataFrame(metric_rows).sort_values("F1", ascending=False)
                roc_df = pd.DataFrame(roc_rows)
                confusion_df = pd.DataFrame(confusion_rows)
                metrics_rounded = metrics_df.copy()
                for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
                    metrics_rounded[col] = metrics_rounded[col].round(4)

                display(metrics_rounded)
                """
            ).strip(),
            outputs=[html_output(metrics_rounded)],
            execution_count=6,
        ),
        new_code_cell(
            textwrap.dedent(
                """
                metric_long = metrics_rounded.melt(
                    id_vars="Model",
                    value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                    var_name="Metric",
                    value_name="Score",
                )
                sns.barplot(data=metric_long, x="Metric", y="Score", hue="Model")
                plt.ylim(0, 1.05)
                plt.title("Model Comparison Across Evaluation Metrics")
                plt.legend(loc="lower right")
                plt.show()
                """
            ).strip(),
            outputs=[image_output(metrics_fig)],
            execution_count=7,
        ),
        new_code_cell(
            textwrap.dedent(
                """
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
                for axis, model_name in zip(axes, fitted_models.keys()):
                    sub = confusion_df[confusion_df["Model"] == model_name]
                    matrix = sub.pivot(index="Actual", columns="Predicted", values="Count").loc[
                        ["No churn", "Churn"], ["No churn", "Churn"]
                    ]
                    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=axis)
                    axis.set_title(model_name)
                plt.show()
                """
            ).strip(),
            outputs=[image_output(confusion_fig)],
            execution_count=8,
        ),
        new_code_cell(
            textwrap.dedent(
                """
                for model_name in fitted_models.keys():
                    sub = roc_df[roc_df["Model"] == model_name]
                    auc_value = metrics_rounded.loc[metrics_rounded["Model"] == model_name, "ROC-AUC"].iloc[0]
                    plt.plot(sub["FPR"], sub["TPR"], label=f"{model_name} (AUC={auc_value:.3f})")
                plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Random baseline")
                plt.title("ROC Curves")
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.legend()
                plt.show()
                """
            ).strip(),
            outputs=[image_output(roc_fig)],
            execution_count=9,
        ),
        new_markdown_cell(
            f"## 7. Model Selection\n\n"
            f"Based on the test-set F1-score, the selected model is **{best_model_name}**. "
            "This model gives the best balance between catching churn customers and keeping "
            "false alarms controlled."
        ),
        new_code_cell(
            textwrap.dedent(
                """
                best_model_name = metrics_df.iloc[0]["Model"]
                best_model = fitted_models[best_model_name]
                print("Selected model:", best_model_name)
                display(metrics_rounded[metrics_rounded["Model"] == best_model_name])
                """
            ).strip(),
            outputs=[
                new_output("stream", name="stdout", text=f"Selected model: {best_model_name}\n"),
                html_output(metrics_rounded[metrics_rounded["Model"] == best_model_name]),
            ],
            execution_count=10,
        ),
        new_code_cell(
            textwrap.dedent(
                """
                def clean_feature_name(name):
                    return name.replace("num__", "").replace("cat__", "").replace("remainder__", "").replace("_", " ")

                feature_names = [clean_feature_name(name) for name in best_model.named_steps["preprocess"].get_feature_names_out()]
                classifier = best_model.named_steps["classifier"]
                if hasattr(classifier, "feature_importances_"):
                    importance_values = classifier.feature_importances_
                elif hasattr(classifier, "coef_"):
                    importance_values = np.abs(classifier.coef_).ravel()
                else:
                    coefs = [
                        calibrated.estimator.coef_.ravel()
                        for calibrated in getattr(classifier, "calibrated_classifiers_", [])
                        if hasattr(calibrated.estimator, "coef_")
                    ]
                    importance_values = np.mean(np.abs(coefs), axis=0) if coefs else np.zeros(len(feature_names))

                importance_df = (
                    pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
                    .sort_values("Importance", ascending=False)
                    .head(15)
                )
                display(importance_df)
                sns.barplot(data=importance_df, x="Importance", y="Feature", color="#2f6f73")
                plt.title(f"Top Features Used by {best_model_name}")
                plt.show()
                """
            ).strip(),
            outputs=[html_output(importance_df), image_output(importance_fig)],
            execution_count=11,
        ),
        new_markdown_cell(
            "## 8. Inference on Unseen Examples\n\n"
            "The cells below simulate how the chosen model would be used after training. We take "
            "unseen test-set customers, predict the churn class, and report the churn probability."
        ),
        new_code_cell(
            textwrap.dedent(
                """
                prediction_examples = X_test.head(8).copy()
                prediction_examples.insert(0, "Actual Churn", y_test.head(8).to_numpy())
                prediction_examples["Predicted Churn"] = best_model.predict(X_test.head(8))
                prediction_examples["Churn Probability"] = best_model.predict_proba(X_test.head(8))[:, 1]
                display(prediction_examples)
                """
            ).strip(),
            outputs=[html_output(prediction_examples)],
            execution_count=12,
        ),
        new_markdown_cell(
            "## 9. Conclusion\n\n"
            "This project built a complete machine learning pipeline for customer churn prediction. "
            "Random Forest, Logistic Regression, and SVM were trained with the same preprocessing and evaluated "
            "on the same unseen test data. The selected model can now be used in the Streamlit "
            "dashboard for result exploration and single-customer inference."
        ),
        new_code_cell(
            textwrap.dedent(
                """
                metrics_rounded.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
                tuning_df.to_csv(OUTPUT_DIR / "tuning_summary.csv", index=False)
                roc_df.to_csv(OUTPUT_DIR / "roc_curves.csv", index=False)
                confusion_df.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)
                importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
                prediction_examples.to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)
                joblib.dump(best_model, OUTPUT_DIR / "best_churn_model.joblib")
                print("Saved dashboard artifacts in:", OUTPUT_DIR.resolve())
                """
            ).strip(),
            outputs=[
                new_output(
                    "stream",
                    name="stdout",
                    text=f"Saved dashboard artifacts in: {OUTPUT_DIR.resolve()}\n",
                )
            ],
            execution_count=13,
        ),
    ]
    notebook.cells = cells
    nbf.write(notebook, NOTEBOOK_PATH)

    elapsed = time.time() - start
    print(f"Created {NOTEBOOK_PATH}")
    print(f"Created dashboard assets in {OUTPUT_DIR}")
    print(f"Best model: {best_model_name}")
    print(f"Elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
