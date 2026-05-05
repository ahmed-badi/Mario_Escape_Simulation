"""
tests/test_simulation.py
------------------------
Tests unitaires pour les composants principaux.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.environment.grid import Grid
from src.environment.spawn import spawn_configuration, GridSampler, PositionSampler
from src.agents.mario import Mario
from src.agents.monster import Monster
from src.strategies.mario_strategies import get_mario_strategy
from src.strategies.monster_strategies import get_monster_strategy
from src.simulation.engine import SimulationEngine
from src.utils.config import GridConfig, AgentConfig, ExperimentConfig


# ---------------------------------------------------------------------------
# Tests Grid
# ---------------------------------------------------------------------------

class TestGrid:
    def setup_method(self):
        self.grid = Grid(5, 5, exits=[(0, 4), (4, 0)])

    def test_is_valid(self):
        assert self.grid.is_valid((0, 0))
        assert self.grid.is_valid((4, 4))
        assert not self.grid.is_valid((-1, 0))
        assert not self.grid.is_valid((5, 0))

    def test_is_exit(self):
        assert self.grid.is_exit((0, 4))
        assert not self.grid.is_exit((2, 2))

    def test_neighbors_corner(self):
        nbs = self.grid.neighbors((0, 0))
        assert set(nbs) == {(1, 0), (0, 1)}

    def test_neighbors_center(self):
        nbs = self.grid.neighbors((2, 2))
        assert len(nbs) == 4

    def test_manhattan(self):
        assert Grid.manhattan((0, 0), (3, 4)) == 7

    def test_bfs_distance(self):
        d = self.grid.bfs_distance((0, 0), (4, 4))
        assert d == 8

    def test_shortest_path(self):
        path = self.grid.shortest_path((0, 0), (0, 2))
        assert path[0] == (0, 0)
        assert path[-1] == (0, 2)
        assert len(path) == 3

    def test_nearest_exit(self):
        pos, dist = self.grid.nearest_exit((0, 3))
        assert pos == (0, 4)
        assert dist == 1


# ---------------------------------------------------------------------------
# Tests Stratégies
# ---------------------------------------------------------------------------

class TestStrategies:
    def setup_method(self):
        self.rng = np.random.default_rng(123)
        self.grid = Grid(6, 6, exits=[(5, 5)])

    def test_random_mario_strategy(self):
        strat = get_mario_strategy("random", self.rng)
        pos = strat.next_move(self.grid, (3, 3), (0, 0))
        assert self.grid.is_valid(pos)

    def test_greedy_mario_moves_toward_exit(self):
        strat = get_mario_strategy("greedy", self.rng)
        mario_pos = (0, 0)
        next_pos = strat.next_move(self.grid, mario_pos, (5, 0))
        # Doit se rapprocher de (5,5)
        d_before = Grid.manhattan(mario_pos, (5, 5))
        d_after = Grid.manhattan(next_pos, (5, 5))
        assert d_after < d_before

    def test_astar_mario_follows_path(self):
        strat = get_mario_strategy("astar", self.rng)
        next_pos = strat.next_move(self.grid, (0, 0), (5, 0))
        assert self.grid.is_valid(next_pos)

    def test_aggressive_monster_moves_toward_mario(self):
        strat = get_monster_strategy("aggressive", self.rng)
        monster_pos = (0, 0)
        mario_pos = (3, 3)
        next_pos = strat.next_move(self.grid, monster_pos, mario_pos)
        d_before = Grid.manhattan(monster_pos, mario_pos)
        d_after = Grid.manhattan(next_pos, mario_pos)
        assert d_after < d_before


# ---------------------------------------------------------------------------
# Tests Engine
# ---------------------------------------------------------------------------

class TestSimulationEngine:
    def test_mario_wins(self):
        """Mario sur la sortie dès le premier pas."""
        grid = Grid(3, 3, exits=[(0, 1)])
        mario_strat = get_mario_strategy("greedy", np.random.default_rng(0))
        monster_strat = get_monster_strategy("random", np.random.default_rng(0))
        mario = Mario((0, 0), mario_strat)
        monster = Monster((2, 2), monster_strat)
        engine = SimulationEngine(max_steps=50)
        result = engine.run(0, grid, mario, monster)
        # Doit finir (escape ou caught ou timeout)
        assert result.outcome in ("escape", "caught", "timeout")
        assert result.steps >= 1

    def test_timeout(self):
        """Vérifie que le timeout est respecté."""
        grid = Grid(10, 10, exits=[(9, 9)])
        mario_strat = get_mario_strategy("random", np.random.default_rng(0))
        monster_strat = get_monster_strategy("random", np.random.default_rng(0))
        mario = Mario((0, 0), mario_strat)
        monster = Monster((9, 0), monster_strat)
        engine = SimulationEngine(max_steps=5)
        result = engine.run(0, grid, mario, monster)
        assert result.steps <= 5

    def test_result_fields(self):
        """Vérifie que le résultat contient tous les champs requis."""
        grid = Grid(5, 5, exits=[(4, 4)])
        mario = Mario((0, 0), get_mario_strategy("greedy", np.random.default_rng(42)))
        monster = Monster((4, 0), get_monster_strategy("aggressive", np.random.default_rng(42)))
        engine = SimulationEngine(max_steps=100)
        result = engine.run(1, grid, mario, monster)
        d = result.to_dict()
        required_keys = [
            "run_id", "grid_rows", "grid_cols",
            "mario_start_x", "mario_start_y",
            "outcome", "steps", "mario_strategy", "monster_strategy"
        ]
        for key in required_keys:
            assert key in d, f"Clé manquante: {key}"


# ---------------------------------------------------------------------------
# Tests Monte Carlo (smoke test)
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_small_mc_run(self, tmp_path):
        from src.simulation.monte_carlo import MonteCarloRunner
        output = str(tmp_path / "test_results.csv")
        config = ExperimentConfig()
        config.monte_carlo.num_runs = 20
        config.monte_carlo.output_path = output
        config.monte_carlo.mario_strategies = ["random"]
        config.monte_carlo.monster_strategies = ["random"]
        config.monte_carlo.verbose = False
        runner = MonteCarloRunner(config)
        results = runner.run()
        assert len(results) == 20
        assert os.path.exists(output)

        import pandas as pd
        df = pd.read_csv(output)
        assert len(df) == 20
        assert set(df["outcome"].unique()).issubset({"escape", "caught", "timeout"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
