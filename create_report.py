from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_DIR / "customer_churn_lab_report.pdf"


def make_table(df: pd.DataFrame, column_widths: list[float] | None = None) -> Table:
    table_data = [df.columns.tolist()] + df.astype(str).values.tolist()
    table = Table(table_data, colWidths=column_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f6f73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d0d7de")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(7.5 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    tuning = pd.read_csv(OUTPUT_DIR / "tuning_summary.csv")
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv").head(10)
    examples = pd.read_csv(OUTPUT_DIR / "sample_predictions.csv").head(5)
    profile = json.loads((OUTPUT_DIR / "data_profile.json").read_text(encoding="utf-8"))

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CenterTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=26,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2f6f73"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=8,
        )
    )

    story = []
    body = styles["Body"]
    heading = styles["SectionTitle"]

    story.append(Paragraph("Customer Churn Prediction", styles["CenterTitle"]))
    story.append(Paragraph("Business / Marketing / Supply Chain Lab Project Report", styles["Heading2"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "This report summarizes a supervised machine learning project for predicting customer churn. "
            "The project compares Random Forest, Logistic Regression, and SVM using a shared preprocessing pipeline, "
            "cross-validation based hyperparameter tuning, and a common unseen test set.",
            body,
        )
    )
    story.append(
        Paragraph(
            f"Dataset: customer churn training data with {profile['clean_rows']:,} clean rows. "
            f"Modeling sample: {profile['sample_rows']:,} stratified rows. Selected model: "
            f"{profile['best_model']}.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("1. Introduction", heading))
    story.append(
        Paragraph(
            "Customer churn is a central business problem because losing active customers reduces revenue "
            "and increases the cost of growth. In marketing and customer success teams, a churn model can "
            "support retention campaigns by ranking customers according to churn risk.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The objective is to predict the binary target Churn, where 1 means the customer churned and "
            "0 means the customer stayed. This is a supervised classification task with measurable outcomes.",
            body,
        )
    )
    story.append(Image(str(FIG_DIR / "target_distribution.png"), width=4.8 * inch, height=3.0 * inch))
    story.append(PageBreak())

    story.append(Paragraph("2. Background", heading))
    story.append(
        Paragraph(
            "The dataset contains demographic, behavioral, contract, spending, and interaction variables. "
            "These variables are relevant because churn is often influenced by customer age, tenure, product "
            "usage, support burden, payment delay, contract length, subscription type, spending, and recency.",
            body,
        )
    )
    feature_df = pd.DataFrame(
        {
            "Feature group": ["Numeric", "Categorical", "Removed identifier", "Target"],
            "Columns": [
                ", ".join(profile["numeric_features"]),
                ", ".join(profile["categorical_features"]),
                "CustomerID",
                "Churn",
            ],
        }
    )
    story.append(make_table(feature_df, [1.5 * inch, 5.2 * inch]))
    story.append(PageBreak())

    story.append(Paragraph("3. Methodology", heading))
    story.append(
        Paragraph(
            "Rows with missing values were removed because the file only contained one incomplete row. "
            "CustomerID was excluded because it is an identifier rather than a meaningful predictor. "
            "Numeric features were median-imputed and scaled. Categorical features were mode-imputed and "
            "one-hot encoded.",
            body,
        )
    )
    story.append(
        Paragraph(
            "A stratified sample was used for modeling so that the notebook remains practical to rerun while "
            "preserving the target distribution. The sampled data was split into 80 percent training and "
            "20 percent testing. All three models used the same split and preprocessing pipeline.",
            body,
        )
    )
    story.append(Paragraph("Hyperparameter Tuning Summary", styles["Heading2"]))
    story.append(make_table(tuning, [1.2 * inch, 1.1 * inch, 4.4 * inch]))
    story.append(PageBreak())

    story.append(Paragraph("4. Algorithms", heading))
    story.append(
        Paragraph(
            "Random Forest is an ensemble of decision trees. It reduces overfitting by averaging many trees "
            "trained on randomized subsets of the data and features. Important hyperparameters include the "
            "number of trees and maximum tree depth.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Logistic Regression is a linear classification model. It estimates the probability of churn by "
            "combining feature values with learned coefficients. Important hyperparameters include the "
            "regularization strength C and the class weighting strategy.",
            body,
        )
    )
    story.append(
        Paragraph(
            "SVM searches for a separating boundary with a maximum margin. A linear SVM was used for speed "
            "and interpretability on the large tabular dataset, with probability calibration for fair ROC-AUC "
            "comparison.",
            body,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("5. Results and Interpretation", heading))
    story.append(
        Paragraph(
            "The comparison uses accuracy, precision, recall, F1-score, and ROC-AUC. F1-score is emphasized "
            "because churn prediction requires a balance between finding true churners and limiting false "
            "retention alarms.",
            body,
        )
    )
    story.append(make_table(metrics, [1.3 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(str(FIG_DIR / "model_comparison.png"), width=6.5 * inch, height=3.4 * inch))
    story.append(PageBreak())

    story.append(Paragraph("6. Model Selection and Inference", heading))
    story.append(
        Paragraph(
            f"The selected model is {profile['best_model']} because it achieved the strongest F1-score on "
            "the unseen test data. The ROC and confusion matrix plots support this choice by showing strong "
            "class separation and a low number of mistakes.",
            body,
        )
    )
    story.append(Image(str(FIG_DIR / "roc_curves.png"), width=5.6 * inch, height=4.0 * inch))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Image(str(FIG_DIR / "confusion_matrices.png"), width=6.4 * inch, height=2.1 * inch))
    story.append(PageBreak())

    story.append(Paragraph("7. Explainability and Conclusion", heading))
    story.append(
        Paragraph(
            "The top feature chart helps explain which customer attributes influenced the selected model most. "
            "This is useful for a business audience because it connects prediction quality to actionable churn "
            "drivers.",
            body,
        )
    )
    story.append(make_table(importance, [4.7 * inch, 1.4 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Example Inference Output", styles["Heading2"]))
    compact_examples = examples[["Actual Churn", "Predicted Churn", "Churn Probability"]].copy()
    compact_examples["Churn Probability"] = compact_examples["Churn Probability"].map(lambda value: f"{value:.2%}")
    story.append(make_table(compact_examples, [1.5 * inch, 1.6 * inch, 1.6 * inch]))
    story.append(
        Paragraph(
            "In conclusion, the project satisfies the core machine learning workflow: clear objective, dataset "
            "preparation, preprocessing, model tuning, fair comparison, final model selection, and inference "
            "on unseen data. The notebook provides the executable workflow, while the Streamlit dashboard "
            "presents the results interactively.",
            body,
        )
    )

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Customer Churn Prediction Lab Report",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Created {REPORT_PATH}")


if __name__ == "__main__":
    main()
