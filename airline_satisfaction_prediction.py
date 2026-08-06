import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

OUT_DIR = "outputs"
EDA_DIR = os.path.join(OUT_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
def load_data():
    if os.path.exists("train.csv"):
        train = pd.read_csv("train.csv")
        if os.path.exists("test.csv"):
            test = pd.read_csv("test.csv")
            df = pd.concat([train, test], ignore_index=True)
        else:
            df = train
    elif os.path.exists("airline_passenger_satisfaction.csv"):
        df = pd.read_csv("airline_passenger_satisfaction.csv")
    else:
        raise FileNotFoundError(
            "No dataset found. Put train.csv (+ optional test.csv) "
            "in this folder, or edit load_data() with your filename."
        )

    # Kaggle version has stray index columns — drop if present
    for col in ["Unnamed: 0", "id"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df


# ---------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------
def clean_data(df):
    print(f"Initial shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Arrival Delay has a few missing values in the real dataset — impute with median
    delay_col = "Arrival Delay in Minutes"
    if delay_col in df.columns:
        df[delay_col].fillna(df[delay_col].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    print(f"Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------
def run_eda(df, target_col):
    # Target distribution
    plt.figure()
    sns.countplot(x=target_col, data=df)
    plt.title("Satisfaction Distribution")
    plt.savefig(os.path.join(EDA_DIR, "target_distribution.png"), bbox_inches="tight")
    plt.close()

    # Satisfaction by Class
    if "Class" in df.columns:
        plt.figure()
        sns.countplot(x="Class", hue=target_col, data=df)
        plt.title("Satisfaction by Travel Class")
        plt.savefig(os.path.join(EDA_DIR, "satisfaction_by_class.png"), bbox_inches="tight")
        plt.close()

    # Satisfaction by Customer Type
    if "Customer Type" in df.columns:
        plt.figure()
        sns.countplot(x="Customer Type", hue=target_col, data=df)
        plt.title("Satisfaction by Customer Type")
        plt.savefig(os.path.join(EDA_DIR, "satisfaction_by_customer_type.png"), bbox_inches="tight")
        plt.close()

    # Age distribution by satisfaction
    if "Age" in df.columns:
        plt.figure()
        sns.histplot(data=df, x="Age", hue=target_col, kde=True, bins=30)
        plt.title("Age Distribution by Satisfaction")
        plt.savefig(os.path.join(EDA_DIR, "age_distribution.png"), bbox_inches="tight")
        plt.close()

    # Correlation heatmap (numeric features only)
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    print(f"EDA plots saved to {EDA_DIR}/")


# ---------------------------------------------------------------------
# 4. PREPROCESSING
# ---------------------------------------------------------------------
def preprocess(df, target_col):
    df = df.copy()

    # Encode target: satisfied -> 1, neutral or dissatisfied -> 0
    df[target_col] = df[target_col].apply(
        lambda x: 1 if str(x).strip().lower() == "satisfied" else 0
    )

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


# ---------------------------------------------------------------------
# 5. MODELING
# ---------------------------------------------------------------------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    report_lines = []

    # --- Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    results["Logistic Regression"] = evaluate_model(
        y_test, y_pred_lr, y_proba_lr, "Logistic Regression", report_lines
    )

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    results["Random Forest"] = evaluate_model(
        y_test, y_pred_rf, y_proba_rf, "Random Forest", report_lines
    )

    # Save report
    with open(os.path.join(OUT_DIR, "model_results.txt"), "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nModel results saved to {OUT_DIR}/model_results.txt")

    # Feature importance (Random Forest)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances.head(15).values, y=importances.head(15).index)
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), bbox_inches="tight")
    plt.close()

    print(f"Feature importance plot saved to {OUT_DIR}/feature_importance.png")
    print("\nTop 10 factors driving satisfaction:")
    print(importances.head(10))

    return results, importances


def evaluate_model(y_test, y_pred, y_proba, name, report_lines):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    block = [
        f"\n{'='*50}",
        f"{name}",
        f"{'='*50}",
        f"Accuracy : {acc:.4f}",
        f"Precision: {prec:.4f}",
        f"Recall   : {rec:.4f}",
        f"F1-score : {f1:.4f}",
        f"ROC-AUC  : {auc:.4f}",
        f"\nConfusion Matrix:\n{cm}",
        f"\nClassification Report:\n{classification_report(y_test, y_pred)}",
    ]
    report_lines.extend(block)
    print("\n".join(block))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ---------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------
def main():
    target_col = "satisfaction"

    df = load_data()
    df = clean_data(df)

    # Handle slightly different column naming across dataset versions
    if target_col not in df.columns:
        for c in df.columns:
            if c.lower().strip() == "satisfaction":
                target_col = c
                break

    run_eda(df, target_col)
    X, y = preprocess(df, target_col)
    results, importances = train_and_evaluate(X, y)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
