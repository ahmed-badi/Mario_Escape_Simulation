"""
src/simulation/monte_carlo.py
------------------------------
Runner Monte Carlo : lance N simulations avec variations
aléatoires de grille, positions et stratégies.

Fonctionnalités :
  - Itération sur toutes les combinaisons de stratégies (cross_strategies)
  - Reproductibilité via seed
  - Export CSV progressif (flush par batch)
  - Barre de progression (tqdm)
"""

from __future__ import annotations
import csv
import os
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from src.utils.config import ExperimentConfig, GridConfig, AgentConfig, SimulationConfig
from src.environment.grid import Grid
from src.environment.spawn import spawn_configuration
from src.agents.mario import Mario
from src.agents.monster import Monster
from src.strategies.mario_strategies import get_mario_strategy
from src.strategies.monster_strategies import get_monster_strategy
from src.simulation.engine import SimulationEngine, SimulationResult


class MonteCarloRunner:
    """
    Orchestre les simulations Monte Carlo massives.

    Usage
    -----
    >>> config = ExperimentConfig()
    >>> runner = MonteCarloRunner(config)
    >>> results = runner.run()
    """

    # Colonnes du CSV dans l'ordre de sortie
    CSV_COLUMNS = [
        "run_id", "grid_rows", "grid_cols",
        "mario_start_x", "mario_start_y",
        "monster_start_x", "monster_start_y",
        "exit_positions",
        "mario_strategy", "monster_strategy",
        "steps", "outcome",
        "mario_path_length",
        "min_dist_to_exit_init",
        "init_mario_monster_dist",
        "mario_path", "monster_path",
    ]

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.mc_cfg = config.monte_carlo
        self.engine = SimulationEngine(
            max_steps=config.simulation.max_steps,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def run(self) -> List[SimulationResult]:
        """Lance toutes les simulations et exporte le CSV."""
        strategy_pairs = self._build_strategy_pairs()
        runs_per_pair = max(1, self.mc_cfg.num_runs // len(strategy_pairs))
        total_runs = runs_per_pair * len(strategy_pairs)

        os.makedirs(os.path.dirname(self.mc_cfg.output_path), exist_ok=True)

        results: List[SimulationResult] = []
        run_id = 0

        with open(self.mc_cfg.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()

            with tqdm(
                total=total_runs,
                desc="Monte Carlo",
                disable=not self.mc_cfg.verbose,
                unit="sim",
            ) as pbar:
                for mario_strat_name, monster_strat_name in strategy_pairs:
                    rng = np.random.default_rng(
                        self.mc_cfg.base_seed + run_id if self.mc_cfg.base_seed else None
                    )

                    for _ in range(runs_per_pair):
                        result = self._run_single(
                            run_id=run_id,
                            mario_strategy_name=mario_strat_name,
                            monster_strategy_name=monster_strat_name,
                            rng=rng,
                        )
                        results.append(result)
                        writer.writerow(result.to_dict())
                        run_id += 1
                        pbar.update(1)

        if self.mc_cfg.verbose:
            self._print_summary(results)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_strategy_pairs(self) -> List[Tuple[str, str]]:
        if self.mc_cfg.cross_strategies:
            return [
                (ms, mns)
                for ms in self.mc_cfg.mario_strategies
                for mns in self.mc_cfg.monster_strategies
            ]
        # Sinon : paires par index (zip)
        return list(zip(
            self.mc_cfg.mario_strategies,
            self.mc_cfg.monster_strategies,
        ))

    def _run_single(
        self,
        run_id: int,
        mario_strategy_name: str,
        monster_strategy_name: str,
        rng: np.random.Generator,
    ) -> SimulationResult:
        """Configure et exécute une simulation individuelle."""

        # --- Générer la configuration spatiale ---
        rows, cols, mario_start, monster_start, exits = spawn_configuration(
            grid_config=self.config.grid,
            agent_config=self.config.agents,
            rng=rng,
        )

        grid = Grid(rows=rows, cols=cols, exits=exits)

        # --- Instancier les agents ---
        mario_strat = get_mario_strategy(mario_strategy_name, rng=rng)
        monster_strat = get_monster_strategy(monster_strategy_name, rng=rng)

        mario = Mario(start_pos=mario_start, strategy=mario_strat)
        monster = Monster(start_pos=monster_start, strategy=monster_strat)

        # --- Lancer la simulation ---
        return self.engine.run(
            run_id=run_id,
            grid=grid,
            mario=mario,
            monster=monster,
        )

    def _print_summary(self, results: List[SimulationResult]) -> None:
        outcomes = [r.outcome for r in results]
        total = len(outcomes)
        escape = outcomes.count("escape")
        caught = outcomes.count("caught")
        timeout = outcomes.count("timeout")
        print(f"\n{'='*50}")
        print(f"  RÉSULTATS MONTE CARLO ({total} simulations)")
        print(f"{'='*50}")
        print(f"  ✅ Escape  : {escape:>6}  ({100*escape/total:5.1f}%)")
        print(f"  ❌ Caught  : {caught:>6}  ({100*caught/total:5.1f}%)")
        print(f"  ⏱️  Timeout : {timeout:>6}  ({100*timeout/total:5.1f}%)")
        print(f"{'='*50}")
        print(f"  📄 CSV exporté → {self.config.monte_carlo.output_path}")
