"""
ml/utils/metrics.py
--------------------
Fonctions d'évaluation communes pour supervisé et RL.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label: str = "",
) -> Dict[str, float]:
    """
    Évalue un classifieur binaire ou multiclasse.
    Retourne un dict de métriques + affiche le rapport.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {"accuracy": acc, "f1_weighted": f1}

    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            metrics["roc_auc"] = auc
        except Exception:
            pass

    header = f"\n{'='*55}\n  Évaluation {label}\n{'='*55}"
    print(header)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (weighted) : {f1:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\n" + classification_report(y_true, y_pred, zero_division=0))
    return metrics


def win_rate(outcomes: List[str]) -> float:
    """Taux d'escape dans une liste d'outcomes."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o == "escape") / len(outcomes)


def strategy_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Construit un DataFrame comparatif des stratégies.
    results = {"strategy_name": {"escape": N, "caught": N, "timeout": N, "mean_steps": X}}
    """
    rows = []
    for name, data in results.items():
        total = data.get("escape", 0) + data.get("caught", 0) + data.get("timeout", 0)
        rows.append({
            "strategy": name,
            "escape_rate": data.get("escape", 0) / max(total, 1),
            "caught_rate": data.get("caught", 0) / max(total, 1),
            "timeout_rate": data.get("timeout", 0) / max(total, 1),
            "mean_steps": data.get("mean_steps", 0),
            "n_episodes": total,
        })
    df = pd.DataFrame(rows).sort_values("escape_rate", ascending=False)
    return df