"""
src/strategies/monster_strategies.py
--------------------------------------
Stratégies de déplacement pour le Monstre.

Stratégies disponibles :
  - RandomMonsterStrategy         : marche aléatoire uniforme
  - AggressiveMonsterStrategy     : poursuite BFS optimale de Mario
  - SemiAggressiveMonsterStrategy : alterne entre poursuite et mouvement aléatoire
                                    selon une probabilité configurable
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.environment.grid import Grid

Position = Tuple[int, int]


class MonsterStrategy(ABC):
    """Interface commune pour toutes les stratégies du Monstre."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def next_move(self, grid: "Grid", monster_pos: Position, mario_pos: Position) -> Position:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Stratégie 1 : Aléatoire
# ---------------------------------------------------------------------------

class RandomMonsterStrategy(MonsterStrategy):
    """Le Monstre se déplace aléatoirement sans cibler Mario."""

    def next_move(self, grid: "Grid", monster_pos: Position, mario_pos: Position) -> Position:
        neighbors = grid.neighbors(monster_pos)
        idx = self.rng.integers(0, len(neighbors))
        return neighbors[idx]


# ---------------------------------------------------------------------------
# Stratégie 2 : Agressive (BFS pursuit)
# ---------------------------------------------------------------------------

class AggressiveMonsterStrategy(MonsterStrategy):
    """
    Le Monstre suit le chemin BFS optimal vers Mario.
    Représente le cas le plus dangereux pour Mario.
    """

    def next_move(self, grid: "Grid", monster_pos: Position, mario_pos: Position) -> Position:
        path = grid.shortest_path(monster_pos, mario_pos)
        # path[0] = monster_pos, path[1] = prochain pas
        if len(path) >= 2:
            return path[1]
        return monster_pos  # déjà sur Mario (fin de partie)


# ---------------------------------------------------------------------------
# Stratégie 3 : Semi-agressive
# ---------------------------------------------------------------------------

class SemiAggressiveMonsterStrategy(MonsterStrategy):
    """
    Le Monstre poursuit Mario avec probabilité `aggression_prob`,
    sinon se déplace aléatoirement.

    Cela simule un ennemi imparfait ou avec des angles morts.
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        aggression_prob: float = 0.7,
    ):
        super().__init__(rng)
        self.aggression_prob = aggression_prob
        self._aggressive = AggressiveMonsterStrategy(rng)
        self._random = RandomMonsterStrategy(rng)

    def next_move(self, grid: "Grid", monster_pos: Position, mario_pos: Position) -> Position:
        if self.rng.random() < self.aggression_prob:
            return self._aggressive.next_move(grid, monster_pos, mario_pos)
        return self._random.next_move(grid, monster_pos, mario_pos)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MONSTER_STRATEGIES = {
    "random": RandomMonsterStrategy,
    "aggressive": AggressiveMonsterStrategy,
    "semi_aggressive": SemiAggressiveMonsterStrategy,
}


def get_monster_strategy(
    name: str, rng: Optional[np.random.Generator] = None
) -> MonsterStrategy:
    if name not in MONSTER_STRATEGIES:
        raise ValueError(
            f"Stratégie Monstre inconnue: '{name}'. Disponibles: {list(MONSTER_STRATEGIES)}"
        )
    return MONSTER_STRATEGIES[name](rng=rng)
