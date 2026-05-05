"""
ml/supervised/train_classifier.py
-----------------------------------
Phase 1 — Entraînement supervisé.

Entraîne plusieurs classifieurs pour prédire si Mario survivra (escape=1)
à partir des features initiales de la simulation.

Modèles entraînés :
  1. Logistic Regression (baseline linéaire)
  2. Random Forest        (ensemble, non-linéaire)
  3. Gradient Boosting   (boosting, souvent le meilleur)

Sorties :
  - data/models/clf_<nom>.pkl pour chaque modèle
  - data/models/best_classifier.pkl (meilleur modèle)
  - data/models/feature_names.json
  - data/processed/classifier_report.csv

Usage :
  python -m ml.supervised.train_classifier
  python -m ml.supervised.train_classifier --csv data/raw/results.csv
  python -m ml.supervised.train_classifier --target multiclass
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

from ml.utils.feature_engineering import load_raw, engineer_features, prepare_Xy
from ml.utils.metrics import evaluate_classifier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ]),
    "random_forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )),
    ]),
    "gradient_boosting": Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=42
        )),
    ]),
}


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model, feature_names, model_name: str, top_n: int = 15):
    clf = model.named_steps.get("clf")
    if clf is None:
        return
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
    else:
        return

    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {top_n} features — {model_name}:")
    for rank, idx in enumerate(indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:<35} {importances[idx]:.4f}")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(model, X, y, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    return {
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std":  float(acc_scores.std()),
        "cv_f1_mean":       float(f1_scores.mean()),
        "cv_f1_std":        float(f1_scores.std()),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(csv_path: str, target: str = "binary", output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)

    # --- Chargement et feature engineering ---
    print(f"\n📂 Chargement : {csv_path}")
    raw = load_raw(csv_path)
    print(f"   {len(raw)} simulations chargées.")

    print("⚙️  Feature engineering...")
    feat_df = engineer_features(raw, include_paths=True)

    target_col = "outcome_binary" if target == "binary" else "outcome"
    X, y, feature_names = prepare_Xy(feat_df, target=target_col)

    print(f"   Features : {X.shape[1]} | Samples : {X.shape[0]}")
    if target == "binary":
        print(f"   Classes  : escape={int(y.sum())} ({100*y.mean():.1f}%) | "
              f"other={int((1-y).sum())} ({100*(1-y).mean():.1f}%)")

    # Sauvegarde des noms de features
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # --- Train / Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train : {len(X_train)} | Test : {len(X_test)}")

    # --- Entraînement + évaluation ---
    results = {}
    best_model_name = None
    best_f1 = -1.0

    for model_name, model in MODELS.items():
        print(f"\n{'─'*55}")
        print(f"🔧 Entraînement : {model_name}")

        # Cross-validation
        cv_metrics = cross_validate_model(model, X_train, y_train, cv=5)
        print(f"   CV Accuracy : {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
        print(f"   CV F1       : {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")

        # Fit sur tout le train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Évaluation test set
        test_metrics = evaluate_classifier(y_test, y_pred, y_proba, label=model_name)

        # Importance des features
        print_feature_importance(model, feature_names, model_name)

        # Sauvegarde du modèle
        model_path = os.path.join(output_dir, f"clf_{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\n   💾 Sauvegardé → {model_path}")

        # Tracking du meilleur
        combined = {**cv_metrics, **test_metrics, "model_name": model_name}
        results[model_name] = combined

        if test_metrics.get("f1_weighted", 0) > best_f1:
            best_f1 = test_metrics["f1_weighted"]
            best_model_name = model_name
            best_model = model

    # --- Sauvegarde du meilleur ---
    best_path = os.path.join(output_dir, "best_classifier.pkl")
    with open(best_path, "wb") as f:
        pickle.dump(best_model, f)

    # --- Rapport CSV ---
    report_df = pd.DataFrame(results).T
    report_path = os.path.join("data", "processed", "classifier_report.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_df.to_csv(report_path)

    print(f"\n{'='*55}")
    print(f"  🏆 Meilleur modèle : {best_model_name}  (F1={best_f1:.4f})")
    print(f"  📄 Rapport       → {report_path}")
    print(f"  💾 Best model    → {best_path}")
    print(f"{'='*55}")

    return best_model, feature_names, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train supervised classifier")
    parser.add_argument("--csv", default="data/raw/results.csv")
    parser.add_argument("--target", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--output-dir", default="data/models")
    args = parser.parse_args()
    train(args.csv, args.target, args.output_dir)