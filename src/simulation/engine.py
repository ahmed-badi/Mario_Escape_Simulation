"""
src/simulation/engine.py
-------------------------
Moteur de simulation pour une seule partie.

Produit un SimulationResult contenant toutes les données
nécessaires pour le dataset (positions initiales, trajectoires,
résultat, métriques).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json

from src.environment.grid import Grid
from src.agents.mario import Mario
from src.agents.monster import Monster

Position = Tuple[int, int]
Outcome = str  # "escape" | "caught" | "timeout"


@dataclass
class SimulationResult:
    """Résultat complet d'une simulation."""

    run_id: int
    grid_rows: int
    grid_cols: int
    mario_start_x: int
    mario_start_y: int
    monster_start_x: int
    monster_start_y: int
    exit_positions: str          # JSON list of [row, col]
    mario_strategy: str
    monster_strategy: str
    steps: int
    outcome: Outcome
    mario_path_length: int
    min_dist_to_exit_init: float
    init_mario_monster_dist: float
    # Trajectoires complètes (optionnel, pour RL)
    mario_path: str = ""         # JSON
    monster_path: str = ""       # JSON

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "mario_start_x": self.mario_start_x,
            "mario_start_y": self.mario_start_y,
            "monster_start_x": self.monster_start_x,
            "monster_start_y": self.monster_start_y,
            "exit_positions": self.exit_positions,
            "mario_strategy": self.mario_strategy,
            "monster_strategy": self.monster_strategy,
            "steps": self.steps,
            "outcome": self.outcome,
            "mario_path_length": self.mario_path_length,
            "min_dist_to_exit_init": self.min_dist_to_exit_init,
            "init_mario_monster_dist": self.init_mario_monster_dist,
            "mario_path": self.mario_path,
            "monster_path": self.monster_path,
        }


class SimulationEngine:
    """
    Exécute une simulation Mario Escape complète.

    Ordre des événements à chaque pas :
      1. Mario choisit et effectue son mouvement
      2. Vérification victoire (Mario sur sortie)
      3. Monstre choisit et effectue son mouvement
      4. Vérification défaite (Monstre sur Mario)
      5. Vérification timeout
    """

    def __init__(self, max_steps: int = 200, verbose: bool = False):
        self.max_steps = max_steps
        self.verbose = verbose

    def run(
        self,
        run_id: int,
        grid: Grid,
        mario: Mario,
        monster: Monster,
    ) -> SimulationResult:
        """
        Lance une simulation et retourne le résultat.
        Les agents doivent déjà être initialisés à leur position de départ.
        """
        mario_start = mario.position
        monster_start = monster.position

        # Métriques initiales
        _, min_dist_exit = grid.nearest_exit(mario_start)
        init_mm_dist = float(grid.manhattan(mario_start, monster_start))

        outcome: Outcome = "timeout"
        steps = 0

        if self.verbose:
            print(f"\n=== Run {run_id} | Grid {grid.rows}x{grid.cols} ===")
            print(f"Mario: {mario_start} | Monster: {monster_start}")
            print(f"Exits: {list(grid.exits)}")
            print(grid.render(mario.position, monster.position))

        for step in range(1, self.max_steps + 1):
            steps = step

            # --- Mario bouge ---
            mario_next = mario.choose_action(grid=grid, monster_pos=monster.position)
            mario.move(mario_next)

            if self.verbose:
                print(f"Step {step}: Mario → {mario.position} | Monster → {monster.position}")

            # --- Victoire : Mario atteint une sortie ---
            if grid.is_exit(mario.position):
                outcome = "escape"
                break

            # --- Monstre bouge ---
            monster_next = monster.choose_action(grid=grid, mario_pos=mario.position)
            monster.move(monster_next)

            # --- Défaite : Monstre attrape Mario ---
            if monster.position == mario.position:
                outcome = "caught"
                break

        if self.verbose:
            print(f"→ Outcome: {outcome} in {steps} steps")
            print(grid.render(mario.position, monster.position))

        return SimulationResult(
            run_id=run_id,
            grid_rows=grid.rows,
            grid_cols=grid.cols,
            mario_start_x=mario_start[0],
            mario_start_y=mario_start[1],
            monster_start_x=monster_start[0],
            monster_start_y=monster_start[1],
            exit_positions=json.dumps(list(grid.exits)),
            mario_strategy=mario.strategy.name,
            monster_strategy=monster.strategy.name,
            steps=steps,
            outcome=outcome,
            mario_path_length=mario.steps_taken,
            min_dist_to_exit_init=float(min_dist_exit),
            init_mario_monster_dist=init_mm_dist,
            mario_path=json.dumps(mario.path),
            monster_path=json.dumps(monster.path),
        )
