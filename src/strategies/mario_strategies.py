"""
src/strategies/mario_strategies.py
------------------------------------
Stratégies de déplacement pour Mario.

Stratégies disponibles :
  - RandomStrategy      : déplacement aléatoire uniforme parmi les voisins
  - GreedyStrategy      : se rapproche de la sortie la plus proche (Manhattan)
  - AStarStrategy       : suit le chemin optimal BFS vers la sortie la plus proche,
                          en évitant le monstre si possible
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.environment.grid import Grid

Position = Tuple[int, int]


class MarioStrategy(ABC):
    """Interface commune pour toutes les stratégies de Mario."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def next_move(self, grid: "Grid", mario_pos: Position, monster_pos: Position) -> Position:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Stratégie 1 : Aléatoire
# ---------------------------------------------------------------------------

class RandomStrategy(MarioStrategy):
    """
    Mario choisit un voisin valide au hasard (marche aléatoire uniforme).
    Sert de baseline / contrôle dans les expériences.
    """

    def next_move(self, grid: "Grid", mario_pos: Position, monster_pos: Position) -> Position:
        neighbors = grid.neighbors(mario_pos)
        idx = self.rng.integers(0, len(neighbors))
        return neighbors[idx]


# ---------------------------------------------------------------------------
# Stratégie 2 : Greedy (heuristique Manhattan)
# ---------------------------------------------------------------------------

class GreedyStrategy(MarioStrategy):
    """
    Mario se dirige vers la case voisine qui minimise la distance Manhattan
    à la sortie la plus proche.
    En cas d'égalité, choisit aléatoirement parmi les ex-aequo.
    """

    def next_move(self, grid: "Grid", mario_pos: Position, monster_pos: Position) -> Position:
        neighbors = grid.neighbors(mario_pos)
        nearest_exit, _ = grid.nearest_exit(mario_pos)

        if nearest_exit is None:
            # Fallback : aléatoire
            idx = self.rng.integers(0, len(neighbors))
            return neighbors[idx]

        best_dist = float("inf")
        best_moves: List[Position] = []

        for nb in neighbors:
            d = grid.manhattan(nb, nearest_exit)
            if d < best_dist:
                best_dist = d
                best_moves = [nb]
            elif d == best_dist:
                best_moves.append(nb)

        idx = self.rng.integers(0, len(best_moves))
        return best_moves[idx]


# ---------------------------------------------------------------------------
# Stratégie 3 : A* / BFS optimal avec évitement du monstre
# ---------------------------------------------------------------------------

class AStarStrategy(MarioStrategy):
    """
    Mario suit le chemin BFS optimal vers la sortie la plus proche.

    Amélioration : si le prochain pas du chemin optimal rapproche
    dangereusement le Monstre, Mario choisit un mouvement de repli
    (celui qui maximise la distance au Monstre tout en ne s'éloignant
    pas trop de la sortie).

    Paramètre danger_radius : si le Monstre est à ≤ danger_radius cases,
    on active la logique d'évitement.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None, danger_radius: int = 2):
        super().__init__(rng)
        self.danger_radius = danger_radius
        self._cached_path: List[Position] = []
        self._path_target: Optional[Position] = None

    def next_move(self, grid: "Grid", mario_pos: Position, monster_pos: Position) -> Position:
        nearest_exit, _ = grid.nearest_exit(mario_pos)
        if nearest_exit is None:
            return mario_pos

        # Recalcul si la cible a changé ou chemin épuisé
        if nearest_exit != self._path_target or not self._cached_path:
            self._cached_path = grid.shortest_path(mario_pos, nearest_exit)
            self._path_target = nearest_exit

        # Supprime les positions déjà passées du chemin
        if self._cached_path and self._cached_path[0] == mario_pos:
            self._cached_path.pop(0)

        if not self._cached_path:
            return mario_pos

        optimal_next = self._cached_path[0]

        # --- Logique d'évitement ---
        monster_dist = grid.manhattan(mario_pos, monster_pos)
        if monster_dist <= self.danger_radius:
            neighbors = grid.neighbors(mario_pos)
            # Score = distance_monstre - 0.5 * distance_sortie (pondération)
            def score(pos: Position) -> float:
                d_monster = grid.manhattan(pos, monster_pos)
                d_exit = grid.manhattan(pos, nearest_exit)
                return d_monster - 0.3 * d_exit

            safe_moves = [nb for nb in neighbors if nb != monster_pos]
            if safe_moves:
                best = max(safe_moves, key=score)
                # Invalider le cache pour le prochain pas
                self._cached_path = []
                return best

        return optimal_next


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MARIO_STRATEGIES = {
    "random": RandomStrategy,
    "greedy": GreedyStrategy,
    "astar": AStarStrategy,
}


def get_mario_strategy(name: str, rng: Optional[np.random.Generator] = None) -> MarioStrategy:
    if name not in MARIO_STRATEGIES:
        raise ValueError(f"Stratégie Mario inconnue: '{name}'. Disponibles: {list(MARIO_STRATEGIES)}")
    return MARIO_STRATEGIES[name](rng=rng)
