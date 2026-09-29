"""
Breast Cancer Predictor
=======================
Trains and compares several classifiers on the Wisconsin Diagnostic Breast
Cancer dataset (bundled with scikit-learn, no download needed), picks the best
one via cross-validation, evaluates it on a held-out test set, saves it, and
provides a predict function.

Usage:
    python breast_cancer_predictor.py            # train, evaluate, save model
    python breast_cancer_predictor.py --demo     # also predict on sample patients

Requirements:
    pip install scikit-learn pandas numpy matplotlib joblib

DISCLAIMER: Educational project only. NOT a medical device and must not be
used for real diagnosis.
"""

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
MODEL_PATH = "breast_cancer_model.joblib"

# In this dataset: 0 = malignant, 1 = benign
LABELS = {0: "Malignant", 1: "Benign"}


def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    counts = y.value_counts().rename(LABELS)
    print(f"Class balance:\n{counts.to_string()}\n")
    return X, y, list(data.feature_names)


def build_candidates():
    """Every model is wrapped in a pipeline so scaling is fit only on training folds."""
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scale", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def select_best_model(candidates, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    print("5-fold cross-validation on training set (ROC AUC):")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = scores.mean()
        print(f"  {name:<22} {scores.mean():.4f} (+/- {scores.std():.4f})")
    best = max(results, key=results.get)
    print(f"\nBest model: {best}\n")
    return best, candidates[best]


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Test set results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, y_prob):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Malignant cases missed (called benign): {cm[0, 1]}")
    print(f"Benign cases flagged as malignant:      {cm[1, 0]}\n")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Malignant", "Benign"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150)
    plt.close()
    print("Saved plot: evaluation.png")


def plot_feature_importance(model, feature_names, X_test, y_test):
    """Model-agnostic permutation importance (works for any classifier)."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test, n_repeats=15, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    imp = pd.Series(result.importances_mean, index=feature_names).sort_values()
    top = imp.tail(10)

    plt.figure(figsize=(8, 5))
    top.plot(kind="barh", color="steelblue")
    plt.xlabel("Drop in ROC AUC when feature is shuffled")
    plt.title("Top 10 Most Important Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved plot: feature_importance.png")
    print("Top 5 features:", ", ".join(imp.tail(5).index[::-1]), "\n")


def predict(features, model_path=MODEL_PATH):
    """
    Predict for one or more patients.

    features: dict {feature_name: value}, list of dicts, or DataFrame containing
              all 30 feature columns.
    Returns a list of dicts with the label and probability of malignancy.
    """
    saved = joblib.load(model_path)
    model, names = saved["model"], saved["feature_names"]

    df = pd.DataFrame([features] if isinstance(features, dict) else features)
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    df = df[names]

    prob_benign = model.predict_proba(df)[:, 1]
    preds = model.predict(df)
    return [
        {
            "prediction": LABELS[int(p)],
            "probability_malignant": round(float(1 - pb), 4),
            "probability_benign": round(float(pb), 4),
        }
        for p, pb in zip(preds, prob_benign)
    ]


def demo(X, y):
    print("=== Demo predictions on 4 random samples ===")
    sample = X.sample(4, random_state=7)
    results = predict(sample)
    for (idx, row), res in zip(sample.iterrows(), results):
        actual = LABELS[int(y.loc[idx])]
        print(
            f"Sample #{idx}: predicted {res['prediction']:<9} "
            f"(P(malignant)={res['probability_malignant']:.3f})  actual: {actual}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run sample predictions")
    args = parser.parse_args()

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    best_name, model = select_best_model(build_candidates(), X_train, y_train)
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)
    plot_feature_importance(model, feature_names, X_test, y_test)

    joblib.dump({"model": model, "feature_names": feature_names, "name": best_name}, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}\n")

    if args.demo:
        demo(X, y)

    print("\nReminder: educational use only, not for medical diagnosis.")


if __name__ == "__main__":
    main()
