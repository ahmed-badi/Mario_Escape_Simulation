"""
src/agents/base_agent.py
------------------------
Classe abstraite de base pour tous les agents de la simulation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List

Position = Tuple[int, int]


class BaseAgent(ABC):
    """
    Interface commune à Mario et au Monstre.

    Chaque agent connaît sa position courante et peut
    calculer son prochain mouvement via `choose_action`.
    """

    def __init__(self, name: str, start_pos: Position):
        self.name = name
        self.position: Position = start_pos
        self.path: List[Position] = [start_pos]

    def move(self, new_pos: Position) -> None:
        self.position = new_pos
        self.path.append(new_pos)

    @abstractmethod
    def choose_action(self, **kwargs) -> Position:
        """
        Retourne la prochaine position souhaitée.
        Les kwargs contiennent le contexte (grille, positions adversaires, etc.)
        """
        ...

    @property
    def steps_taken(self) -> int:
        return len(self.path) - 1

    def reset(self, start_pos: Position) -> None:
        self.position = start_pos
        self.path = [start_pos]

    def __repr__(self) -> str:
        return f"{self.name}@{self.position}"
