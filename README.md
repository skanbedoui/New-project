# Customer Churn Prediction using Machine Learning

CS280 / CS485 Artificial Intelligence lab project for supervised binary classification of customer churn.

## Project Overview

This project predicts whether a customer is likely to churn using demographic, subscription, usage, support, payment, and spending features. The target variable is `Churn`, where `1` means the customer churned and `0` means the customer stayed.

Customer churn prediction is useful for business retention planning because it helps identify customers who may need support, loyalty offers, or other intervention before revenue is lost. This is a binary classification problem.

## Required Models

The project compares exactly three supervised machine learning models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

The final model is selected using business-aware metrics, especially F1-score and recall. In this context, a false negative means the model misses a real churner, which can be costly because the business loses the chance to intervene.

## Dataset

Dataset source:

[Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download)

Download the dataset from Kaggle and place the CSV files in the local `dataset/` folder:

```text
dataset/customer_churn_dataset-training-master.csv
dataset/customer_churn_dataset-testing-master.csv
```

The notebook and dashboard use relative paths, so no machine-specific paths are required. The `dataset/` folder is intentionally ignored by git because dataset files are large/local files.

## Machine Learning Workflow

The notebook includes the complete workflow required for the lab:

- Dataset loading from `dataset/`
- Data inspection, missing-value checks, duplicates, and target distribution
- Data cleaning, including removal of malformed target rows
- Exploratory data analysis with plots and interpretations
- Leakage-safe preprocessing using scikit-learn `Pipeline` and `ColumnTransformer`
- Numerical imputation and scaling
- Categorical imputation and one-hot encoding with `handle_unknown="ignore"`
- Untuned baseline models
- Hyperparameter tuning with cross-validation
- Fair final comparison on unseen testing data
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, classification reports, and ROC curves
- Final model selection based on business meaning, not accuracy alone
- Inference on new customer profiles
- Saved dashboard-ready model artifact

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

On Windows PowerShell, activation may be:

```powershell
.\.venv\Scripts\Activate.ps1
```

The `.venv/` folder is intentionally ignored by git.

## Run the Notebook

```bash
jupyter notebook customer_churn_lab_project.ipynb
```

Run all cells from top to bottom. The notebook generates or updates:

- `models/churn_prediction_model.pkl`
- `outputs/baseline_metrics.csv`
- `outputs/model_metrics.csv`
- `outputs/tuning_summary.csv`
- `outputs/roc_curves.csv`
- `outputs/confusion_matrices.csv`
- `outputs/feature_importance.csv`
- `outputs/sample_predictions.csv`
- `outputs/data_profile.json`
- `outputs/figures/`

The saved model artifact is a full preprocessing-plus-model pipeline, not only a bare classifier.

## Run the Dashboard

After running the notebook once:

```bash
streamlit run streamlit_dashboard.py
```

The Streamlit dashboard loads `models/churn_prediction_model.pkl` and supports:

- Project overview
- Final model metrics
- Model comparison charts
- ROC curve display
- Confusion matrix display
- Dataset preview
- Single-customer churn prediction

## Report

The polished Word report is:

```text
customer_churn_lab_report.docx
```

It summarizes the project objective, dataset, methodology, required models, tuning results, final evaluation, selected model, inference examples, dashboard artifacts, limitations, and future work.

## Project Structure

```text
.
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- customer_churn_lab_project.ipynb
|-- customer_churn_lab_report.docx
|-- streamlit_dashboard.py
|-- dataset/
|   |-- customer_churn_dataset-training-master.csv
|   `-- customer_churn_dataset-testing-master.csv
|-- models/
|   `-- churn_prediction_model.pkl
`-- outputs/
    |-- baseline_metrics.csv
    |-- model_metrics.csv
    |-- tuning_summary.csv
    |-- roc_curves.csv
    |-- confusion_matrices.csv
    |-- feature_importance.csv
    |-- sample_predictions.csv
    |-- data_profile.json
    `-- figures/
```

## Git Notes

The following are intentionally ignored:

- `.venv/`
- `dataset/`
- Python cache files
- Notebook checkpoints

This keeps the repository clean while allowing the project to run locally after downloading the dataset from Kaggle.
