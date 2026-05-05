"""
src/agents/mario.py
-------------------
Agent Mario. Délègue le choix d'action à une stratégie externe
(pattern Strategy), ce qui permet de tester plusieurs comportements
sans modifier la classe agent.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from src.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from src.environment.grid import Grid
    from src.strategies.mario_strategies import MarioStrategy

Position = Tuple[int, int]


class Mario(BaseAgent):

    def __init__(self, start_pos: Position, strategy: "MarioStrategy"):
        super().__init__("Mario", start_pos)
        self.strategy = strategy

    def choose_action(self, grid: "Grid", monster_pos: Position) -> Position:
        return self.strategy.next_move(
            grid=grid,
            mario_pos=self.position,
            monster_pos=monster_pos,
        )
