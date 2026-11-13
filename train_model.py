"""
Train EV battery health classifiers using the official Battery_RUL_Dataset.csv.
SMOTE is inside imblearn Pipeline, so it is applied only to training folds.
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

TARGET = "Battery_Health_Class"
NUMERIC = [
    "Voltage_V", "Current_A", "Temperature_C", "SOH_Percentage",
    "Internal_Resistance_mOhm", "Cycle_Count", "Impedance_Growth", "Aging_Rate",
    "Power_W", "SOH_per_Cycle", "Resistance_SOH_Ratio", "Thermal_Stress", "Aging_Impact"
]
CATEGORICAL = [
    "Vehicle_Type", "Battery_Chemistry", "Manufacturer",
    "Driving_Load", "Weather_Condition"
]
OUTLIER_COLS = [
    "Voltage_V", "Current_A", "Temperature_C",
    "SOH_Percentage", "Internal_Resistance_mOhm"
]
CLASSES = ["Healthy", "Moderate", "Degraded", "Critical"]

def feature_engineering(df):
    df = df.copy()
    required = ["Voltage_V","Current_A","Temperature_C","SOH_Percentage",
                "Internal_Resistance_mOhm","Cycle_Count","Impedance_Growth","Aging_Rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["Power_W"] = df["Voltage_V"] * df["Current_A"]
    df["SOH_per_Cycle"] = df["SOH_Percentage"] / (df["Cycle_Count"] + 1)
    df["Resistance_SOH_Ratio"] = df["Internal_Resistance_mOhm"] / (df["SOH_Percentage"] + 1)
    df["Thermal_Stress"] = df["Temperature_C"] * df["Current_A"] / 100
    df["Aging_Impact"] = df["Impedance_Growth"] * df["Aging_Rate"] * 100
    return df

def remove_training_iqr_outliers(X, y):
    """IQR bounds are learned from training data only; test data is never filtered."""
    keep = pd.Series(True, index=X.index)
    for col in OUTLIER_COLS:
        q1, q3 = X[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep &= X[col].between(lo, hi) | X[col].isna()
    return X.loc[keep].copy(), y.loc[keep].copy()

def make_preprocessor(num_cols, cat_cols):
    num_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, num_cols),
        ("categorical", cat_pipe, cat_cols)
    ], remainder="drop")

def save_eda(df, y):
    plt.figure(figsize=(7,5))
    y.value_counts().reindex(CLASSES).plot(kind="bar")
    plt.title("Battery Health Class Distribution"); plt.xlabel("Health class"); plt.ylabel("Records")
    plt.tight_layout(); plt.savefig(FIG/"01_target_distribution.png", dpi=160); plt.close()

    for col in ["SOH_Percentage","Temperature_C","Voltage_V","Internal_Resistance_mOhm"]:
        plt.figure(figsize=(7,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}"); plt.tight_layout()
        plt.savefig(FIG/f"distribution_{col}.png", dpi=160); plt.close()

    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12,9))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation Heatmap"); plt.tight_layout()
    plt.savefig(FIG/"06_correlation_heatmap.png", dpi=160); plt.close()

def evaluate_model(name, model, X_train, y_train, X_test, y_test, cv):
    cv_result = cross_validate(model, X_train, y_train, cv=cv,
                               scoring="f1_weighted", n_jobs=-1,
                               return_train_score=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    labels = model.named_steps["classifier"].classes_
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "cv_f1_weighted_mean": float(np.mean(cv_result["test_score"])),
        "cv_f1_weighted_std": float(np.std(cv_result["test_score"])),
        "train_f1_weighted_mean": float(np.mean(cv_result["train_score"])),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
    }
    try:
        y_bin = label_binarize(y_test, classes=labels)
        metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(
            y_bin, prob, multi_class="ovr", average="weighted"
        ))
    except Exception as exc:
        warnings.warn(f"ROC-AUC unavailable for {name}: {exc}")
        metrics["roc_auc_ovr_weighted"] = None

    cm = confusion_matrix(y_test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=30)
    plt.title(f"{name} Confusion Matrix"); plt.tight_layout()
    plt.savefig(FIG/f"{name.lower().replace(' ','_')}_confusion_matrix.png", dpi=160); plt.close()

    return model, metrics, pred, prob, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Battery_RUL_Dataset.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    print("Loaded:", df.shape)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Found columns: {list(df.columns)}")
    df = feature_engineering(df)
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(str)
    X = df.drop(columns=[TARGET])
    num_cols = [c for c in NUMERIC if c in X.columns]
    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    missing_cats = [c for c in CATEGORICAL if c not in cat_cols]
    if missing_cats:
        raise ValueError(f"Missing categorical columns: {missing_cats}")
    missing_nums = [c for c in NUMERIC if c not in num_cols]
    if missing_nums:
        raise ValueError(f"Missing numeric columns: {missing_nums}")

    # Hold out the test set before fitting any preprocessing or resampling.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # IQR outlier filtering is applied to training rows only to avoid test leakage.
    X_train, y_train = remove_training_iqr_outliers(X_train, y_train)
    save_eda(df, y)

    pre = make_preprocessor(num_cols, cat_cols)
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", class_weight=None
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics, fitted = {}, {}
    for name, clf in models.items():
        pipe = ImbPipeline([
            ("preprocess", pre),
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("classifier", clf)
        ])
        fitted[name], all_metrics[name], _, _, _ = evaluate_model(
            name, pipe, X_train, y_train, X_test, y_test, cv
        )
        joblib.dump(fitted[name], MODELS / f"{name.lower().replace(' ','_')}.joblib")
        print(name, all_metrics[name]["accuracy"], all_metrics[name]["f1_weighted"])

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1_weighted"])
    joblib.dump({"model": fitted[best_name], "model_name": best_name,
                 "classes": list(fitted[best_name].named_steps["classifier"].classes_),
                 "numeric": num_cols, "categorical": cat_cols},
                MODELS/"best_model_bundle.joblib")

    # Metric comparison chart
    metric_names = ["accuracy","precision_weighted","recall_weighted","f1_weighted"]
    plot_df = pd.DataFrame({n:{m:all_metrics[n][m] for m in metric_names} for n in all_metrics}).T
    plot_df.plot(kind="bar", figsize=(9,5), ylim=(0,1))
    plt.title("Model Performance Comparison"); plt.ylabel("Score")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(FIG/"08_model_performance_comparison.png", dpi=160); plt.close()

    results = {
        "dataset_rows_original": int(len(df)),
        "train_rows_after_iqr": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_classes": sorted(y.unique().tolist()),
        "best_model": best_name,
        "models": all_metrics
    }
    (OUT/"metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved metrics to", OUT/"metrics.json")
    print("Best model:", best_name)

if __name__ == "__main__":
    main()
