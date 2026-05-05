"""
src/analysis/stats.py
----------------------
Analyse statistique des résultats de simulation.

Fonctions disponibles :
  - survival_rates()         : taux par stratégie et outcome
  - summary_by_strategy()    : DataFrame agrégé
  - confidence_interval()    : IC Wilson pour proportions
  - plot_survival_rates()    : graphique comparatif
  - plot_steps_distribution(): distribution du nombre de pas
  - plot_heatmap_outcomes()  : carte de chaleur outcome vs stratégies
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["outcome"] = df["outcome"].astype("category")
    return df


# ---------------------------------------------------------------------------
# Statistiques de base
# ---------------------------------------------------------------------------

def survival_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les taux escape / caught / timeout
    par combinaison (mario_strategy, monster_strategy).
    """
    grouped = (
        df.groupby(["mario_strategy", "monster_strategy", "outcome"])
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby(["mario_strategy", "monster_strategy"]).size().reset_index(name="total")
    merged = grouped.merge(totals, on=["mario_strategy", "monster_strategy"])
    merged["rate"] = merged["count"] / merged["total"]
    return merged


def summary_by_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame résumé avec une ligne par combinaison de stratégies.
    Colonnes : escape_rate, caught_rate, timeout_rate, mean_steps, std_steps.
    """
    def agg_group(g):
        total = len(g)
        return pd.Series({
            "n_simulations": total,
            "escape_rate": (g["outcome"] == "escape").mean(),
            "caught_rate": (g["outcome"] == "caught").mean(),
            "timeout_rate": (g["outcome"] == "timeout").mean(),
            "mean_steps": g["steps"].mean(),
            "std_steps": g["steps"].std(),
            "mean_init_mm_dist": g["init_mario_monster_dist"].mean(),
            "mean_dist_to_exit": g["min_dist_to_exit_init"].mean(),
        })

    return (
        df.groupby(["mario_strategy", "monster_strategy"])
        .apply(agg_group)
        .reset_index()
    )


def confidence_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Intervalle de confiance Wilson pour une proportion.
    Retourne (lower, upper).
    """
    from scipy import stats
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total if total > 0 else 0
    n = total
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_survival_rates(df: pd.DataFrame, save_path: Optional[str] = None):
    """Graphique en barres groupées : taux de survie par stratégie."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    summary = summary_by_strategy(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(summary))
    width = 0.25
    labels = [
        f"{row.mario_strategy}\nvs {row.monster_strategy}"
        for _, row in summary.iterrows()
    ]

    bars_escape  = ax.bar(x - width, summary["escape_rate"],  width, label="Escape ✅",  color="#2ecc71")
    bars_caught  = ax.bar(x,         summary["caught_rate"],  width, label="Caught ❌",  color="#e74c3c")
    bars_timeout = ax.bar(x + width, summary["timeout_rate"], width, label="Timeout ⏱️", color="#95a5a6")

    ax.set_xlabel("Combinaison de stratégies")
    ax.set_ylabel("Taux")
    ax.set_title("Taux de survie de Mario par stratégie (Monte Carlo)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_steps_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """Histogramme du nombre de pas par outcome."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    outcomes = ["escape", "caught", "timeout"]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

    for ax, outcome, color in zip(axes, outcomes, colors):
        subset = df[df["outcome"] == outcome]["steps"]
        if len(subset) > 0:
            ax.hist(subset, bins=30, color=color, edgecolor="white", alpha=0.8)
        ax.set_title(f"Outcome: {outcome}\n(n={len(subset)})")
        ax.set_xlabel("Nombre de pas")
        ax.set_ylabel("Fréquence")
        ax.grid(alpha=0.3)

    plt.suptitle("Distribution du nombre de pas par résultat", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_heatmap_outcomes(df: pd.DataFrame, save_path: Optional[str] = None):
    """Heatmap du taux d'escape : mario_strategy x monster_strategy."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")

    pivot = df.groupby(["mario_strategy", "monster_strategy"]).apply(
        lambda g: (g["outcome"] == "escape").mean()
    ).reset_index(name="escape_rate")

    matrix = pivot.pivot(
        index="mario_strategy", columns="monster_strategy", values="escape_rate"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, ax=ax, linewidths=0.5
    )
    ax.set_title("Taux d'escape de Mario\n(mario_strategy × monster_strategy)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def print_report(df: pd.DataFrame) -> None:
    """Affiche un rapport textuel complet dans la console."""
    summary = summary_by_strategy(df)
    print("\n" + "="*70)
    print("  RAPPORT D'ANALYSE — MARIO ESCAPE SIMULATION")
    print("="*70)
    print(f"  Total simulations : {len(df)}")
    print(f"  Configurations uniques de stratégies : {len(summary)}")
    print()

    for _, row in summary.iterrows():
        n = int(row["n_simulations"])
        e = int(row["escape_rate"] * n)
        lo, hi = confidence_interval(e, n)
        print(f"  [{row['mario_strategy']}] vs [{row['monster_strategy']}]")
        print(f"    Simulations : {n}")
        print(f"    Escape  : {row['escape_rate']:.2%}  IC95% [{lo:.2%}, {hi:.2%}]")
        print(f"    Caught  : {row['caught_rate']:.2%}")
        print(f"    Timeout : {row['timeout_rate']:.2%}")
        print(f"    Pas moy.: {row['mean_steps']:.1f} ± {row['std_steps']:.1f}")
        print()
    print("="*70)
