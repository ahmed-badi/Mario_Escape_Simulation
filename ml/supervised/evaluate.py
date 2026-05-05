"""
ml/supervised/evaluate.py
--------------------------
Évalue un modèle supervisé sauvegardé sur de nouvelles données.
Génère des visualisations (matrice de confusion, courbe ROC, calibration).

Usage :
  python -m ml.supervised.evaluate
  python -m ml.supervised.evaluate --model data/models/best_classifier.pkl
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from ml.utils.feature_engineering import load_raw, engineer_features, prepare_Xy
from ml.utils.metrics import evaluate_classifier


def evaluate(
    model_path: str = "data/models/best_classifier.pkl",
    csv_path: str = "data/raw/results.csv",
    feature_names_path: str = "data/models/feature_names.json",
    plots_dir: str = "data/processed",
):
    os.makedirs(plots_dir, exist_ok=True)

    # --- Chargement ---
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(feature_names_path) as f:
        feature_names = json.load(f)

    raw = load_raw(csv_path)
    feat_df = engineer_features(raw, include_paths=True)
    X, y, _ = prepare_Xy(feat_df, target="outcome_binary", feature_cols=feature_names)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    evaluate_classifier(y_test, y_pred, y_proba, label=os.path.basename(model_path))

    # --- Plot 1 : Matrice de confusion ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Other", "Escape"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Matrice de confusion")

    # --- Plot 2 : Courbe ROC ---
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color="#2ecc71", lw=2, label=f"AUC = {roc_auc:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[1].set_xlabel("Faux positifs")
        axes[1].set_ylabel("Vrais positifs")
        axes[1].set_title("Courbe ROC")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # --- Plot 3 : Courbe de calibration ---
        fraction_pos, mean_pred = calibration_curve(y_test, y_proba[:, 1], n_bins=10)
        axes[2].plot(mean_pred, fraction_pos, "s-", color="#e74c3c", label="Modèle")
        axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Calibration parfaite")
        axes[2].set_xlabel("Probabilité prédite")
        axes[2].set_ylabel("Fraction positive réelle")
        axes[2].set_title("Courbe de calibration")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.suptitle("Évaluation du classifieur supervisé", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "classifier_evaluation.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n📈 Graphiques sauvegardés → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/best_classifier.pkl")
    parser.add_argument("--csv", default="data/raw/results.csv")
    parser.add_argument("--plots-dir", default="data/processed")
    args = parser.parse_args()
    evaluate(args.model, args.csv, plots_dir=args.plots_dir)