from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "images"

def main():
    for folder in (MODELS_DIR, REPORTS_DIR, IMAGES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download Telco-Customer-Churn.csv and put it inside the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"Churn", "customerID", "TotalCharges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    # Clean numeric TotalCharges; blank strings become missing values.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["Churn", "TotalCharges"]).copy()
    df = df.drop_duplicates().copy()

    # Basic dataset summary.
    summary = {
        "rows_after_cleaning": int(df.shape[0]),
        "columns_including_target": int(df.shape[1]),
        "churn_rate_percent": round(float((df["Churn"] == "Yes").mean() * 100), 2),
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
    }
    (REPORTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    df["Churn"].value_counts().rename_axis("Churn").to_frame("count").to_csv(
        REPORTS_DIR / "churn_distribution.csv"
    )

    # EDA charts.
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_title("Customer Churn Distribution")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "churn_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
    ax.set_title("Churn by Contract Type")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "churn_by_contract.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)
    ax.set_title("Customer Tenure by Churn")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "tenure_by_churn.png", dpi=160)
    plt.close(fig)

    target = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn", "customerID"])
    y = target.astype(int)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    results = []
    trained = {}
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
        }
        results.append(row)
        trained[name] = pipeline
        print(f"\n{name}")
        print(classification_report(y_test, pred, zero_division=0))

    results_df = pd.DataFrame(results).sort_values(
        ["f1", "roc_auc"], ascending=False
    )
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_name = str(results_df.iloc[0]["model"])
    best_model = trained[best_name]
    joblib.dump(best_model, MODELS_DIR / "customer_churn_model.joblib")
    metadata = {
        "selected_model": best_name,
        "selection_rule": "Highest test-set F1; ROC-AUC used as tie-breaker",
        "features": X.columns.tolist(),
        "target_mapping": {"No": 0, "Yes": 1},
        "test_metrics": results_df.iloc[0].to_dict(),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Confusion matrix and ROC curve for selected model.
    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No churn", "Churn"],
                yticklabels=["No churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, best_prob, ax=ax)
    ax.set_title(f"ROC Curve — {best_name}")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    print(f"\nSaved best model: {MODELS_DIR / 'customer_churn_model.joblib'}")
    print(f"Selected model by F1: {best_name}")
    print(f"Reports and charts saved under: {REPORTS_DIR} and {IMAGES_DIR}")

if __name__ == "__main__":
    main()
