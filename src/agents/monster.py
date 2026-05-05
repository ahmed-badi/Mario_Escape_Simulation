"""
src/agents/monster.py
---------------------
Agent Monstre. Même pattern Strategy que Mario.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

from src.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from src.environment.grid import Grid
    from src.strategies.monster_strategies import MonsterStrategy

Position = Tuple[int, int]


class Monster(BaseAgent):

    def __init__(self, start_pos: Position, strategy: "MonsterStrategy"):
        super().__init__("Monster", start_pos)
        self.strategy = strategy

    def choose_action(self, grid: "Grid", mario_pos: Position) -> Position:
        return self.strategy.next_move(
            grid=grid,
            monster_pos=self.position,
            mario_pos=mario_pos,
        )
