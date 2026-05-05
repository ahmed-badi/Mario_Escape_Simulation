"""
ml/utils/feature_engineering.py
---------------------------------
Transforme le CSV brut en features numériques prêtes pour ML.

Features extraites :
  - Spatiales  : distances initiales, positions relatives
  - Topologiques : ratio grille, densité de sorties
  - Stratégiques : encodage one-hot des stratégies
  - Dynamiques : path tortuosity (optionnel, depuis trajectoires JSON)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Mapping stratégies → entier (pour embedding / one-hot)
# ---------------------------------------------------------------------------

MARIO_STRATEGY_MAP = {"RandomStrategy": 0, "GreedyStrategy": 1, "AStarStrategy": 2}
MONSTER_STRATEGY_MAP = {
    "RandomMonsterStrategy": 0,
    "AggressiveMonsterStrategy": 1,
    "SemiAggressiveMonsterStrategy": 2,
}
OUTCOME_MAP = {"escape": 0, "caught": 1, "timeout": 2}
BINARY_OUTCOME_MAP = {"escape": 1, "caught": 0, "timeout": 0}  # pour classification binaire


def load_raw(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def parse_exits(exit_str: str) -> List[List[int]]:
    try:
        return json.loads(exit_str)
    except Exception:
        return []


def path_tortuosity(path_json: str) -> float:
    """
    Ratio entre la longueur réelle du chemin et la distance de déplacement
    nette (distance Manhattan start→end). Mesure l'efficacité du chemin.
    1.0 = chemin parfaitement direct. Plus grand = plus tortueux.
    """
    try:
        path = json.loads(path_json)
        if len(path) < 2:
            return 1.0
        n_steps = len(path) - 1
        start, end = path[0], path[-1]
        manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
        if manhattan == 0:
            return float(n_steps)  # a tourné en rond
        return n_steps / manhattan
    except Exception:
        return 1.0


def nearest_exit_distance(mario_x: int, mario_y: int, exits: List[List[int]]) -> float:
    """Distance Manhattan de Mario à la sortie la plus proche."""
    if not exits:
        return 0.0
    dists = [abs(mario_x - ex[0]) + abs(mario_y - ex[1]) for ex in exits]
    return float(min(dists))


def relative_position_to_exit(
    mario_x: int, mario_y: int, exits: List[List[int]]
) -> Tuple[float, float]:
    """Vecteur directionnel normalisé Mario → sortie la plus proche."""
    if not exits:
        return 0.0, 0.0
    best = min(exits, key=lambda e: abs(mario_x - e[0]) + abs(mario_y - e[1]))
    dx = best[0] - mario_x
    dy = best[1] - mario_y
    norm = max(1, abs(dx) + abs(dy))
    return dx / norm, dy / norm


def engineer_features(df: pd.DataFrame, include_paths: bool = True) -> pd.DataFrame:
    """
    Transforme le DataFrame brut en features ML complètes.

    Paramètres
    ----------
    df           : DataFrame brut issu du CSV
    include_paths: si True, calcule les features de trajectoire (plus lent)

    Retourne
    --------
    DataFrame avec uniquement les features numériques + colonnes cible
    """
    feat = pd.DataFrame()

    # --- Features brutes ---
    feat["grid_rows"] = df["grid_rows"]
    feat["grid_cols"] = df["grid_cols"]
    feat["grid_area"] = df["grid_rows"] * df["grid_cols"]
    feat["grid_ratio"] = df["grid_rows"] / df["grid_cols"].clip(lower=1)

    feat["mario_start_x"] = df["mario_start_x"]
    feat["mario_start_y"] = df["mario_start_y"]
    feat["monster_start_x"] = df["monster_start_x"]
    feat["monster_start_y"] = df["monster_start_y"]

    # Position normalisée sur la grille
    feat["mario_x_norm"] = df["mario_start_x"] / df["grid_rows"].clip(lower=1)
    feat["mario_y_norm"] = df["mario_start_y"] / df["grid_cols"].clip(lower=1)
    feat["monster_x_norm"] = df["monster_start_x"] / df["grid_rows"].clip(lower=1)
    feat["monster_y_norm"] = df["monster_start_y"] / df["grid_cols"].clip(lower=1)

    # --- Features de distance ---
    feat["init_mario_monster_dist"] = df["init_mario_monster_dist"]
    feat["min_dist_to_exit_init"] = df["min_dist_to_exit_init"]

    # Distance normalisée par la taille de grille
    diag = np.sqrt(df["grid_rows"] ** 2 + df["grid_cols"] ** 2).clip(lower=1)
    feat["mario_monster_dist_norm"] = df["init_mario_monster_dist"] / diag
    feat["dist_to_exit_norm"] = df["min_dist_to_exit_init"] / diag

    # Ratio : Mario est-il plus proche de la sortie que du Monstre ?
    feat["exit_vs_monster_ratio"] = (
        df["min_dist_to_exit_init"] / df["init_mario_monster_dist"].clip(lower=0.1)
    )

    # --- Exits ---
    exits_parsed = df["exit_positions"].apply(parse_exits)

    feat["num_exits"] = exits_parsed.apply(len)
    feat["exit_density"] = feat["num_exits"] / feat["grid_area"].clip(lower=1)

    # Vecteur directionnel Mario → sortie la plus proche
    dir_feats = [
        relative_position_to_exit(row.mario_start_x, row.mario_start_y, exits_parsed[i])
        for i, row in df.iterrows()
    ]
    feat["exit_dir_x"] = [d[0] for d in dir_feats]
    feat["exit_dir_y"] = [d[1] for d in dir_feats]

    # Vecteur directionnel Monstre → Mario
    feat["monster_to_mario_x"] = (df["mario_start_x"] - df["monster_start_x"]) / df["grid_rows"].clip(lower=1)
    feat["monster_to_mario_y"] = (df["mario_start_y"] - df["monster_start_y"]) / df["grid_cols"].clip(lower=1)

    # --- Stratégies (one-hot) ---
    mario_ohe = pd.get_dummies(df["mario_strategy"], prefix="mario_strat")
    monster_ohe = pd.get_dummies(df["monster_strategy"], prefix="monster_strat")
    feat = pd.concat([feat, mario_ohe, monster_ohe], axis=1)

    # Encodage entier aussi (utile pour certains modèles)
    feat["mario_strategy_id"] = df["mario_strategy"].map(MARIO_STRATEGY_MAP).fillna(-1)
    feat["monster_strategy_id"] = df["monster_strategy"].map(MONSTER_STRATEGY_MAP).fillna(-1)

    # --- Features de trajectoire (optionnel) ---
    if include_paths and "mario_path" in df.columns:
        feat["mario_tortuosity"] = df["mario_path"].apply(path_tortuosity)
        feat["monster_tortuosity"] = df["monster_path"].apply(path_tortuosity)

    # --- Cibles ---
    feat["outcome"] = df["outcome"].map(OUTCOME_MAP)
    feat["outcome_binary"] = df["outcome"].map(BINARY_OUTCOME_MAP)
    feat["steps"] = df["steps"]
    feat["outcome_raw"] = df["outcome"]

    return feat


def get_feature_columns(feat_df: pd.DataFrame) -> List[str]:
    """Retourne la liste des colonnes features (exclut les cibles)."""
    exclude = {"outcome", "outcome_binary", "steps", "outcome_raw"}
    return [c for c in feat_df.columns if c not in exclude]


def prepare_Xy(
    feat_df: pd.DataFrame,
    target: str = "outcome_binary",
    feature_cols: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Retourne (X, y, feature_names) prêts pour scikit-learn.
    target = 'outcome_binary' (0/1) ou 'outcome' (0/1/2 multiclasse)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(feat_df)
    X = feat_df[feature_cols].values.astype(np.float32)
    y = feat_df[target].values
    return X, y, feature_cols