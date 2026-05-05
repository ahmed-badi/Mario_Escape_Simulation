"""
src/environment/spawn.py
------------------------
Générateurs de positions et tailles de grille pour Monte Carlo.
Supporte plusieurs lois de distribution pour varier les configurations.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from src.utils.config import GridConfig, AgentConfig, PositionSamplingMode


Position = Tuple[int, int]


class GridSampler:
    """
    Génère des tailles de grille aléatoires selon différentes lois.
    """

    def __init__(self, config: GridConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def sample_size(self) -> Tuple[int, int]:
        """Retourne (rows, cols) selon le mode configuré."""
        mode = self.config.sampling_mode
        if mode == "fixed":
            return self.config.rows, self.config.cols

        elif mode == "uniform":
            rows = int(self.rng.integers(self.config.rows_min, self.config.rows_max + 1))
            cols = int(self.rng.integers(self.config.cols_min, self.config.cols_max + 1))
            return rows, cols

        elif mode == "normal":
            lo, hi = self.config.size_min, self.config.size_max
            rows = int(np.clip(
                self.rng.normal(self.config.rows_mu, self.config.rows_sigma),
                lo, hi
            ))
            cols = int(np.clip(
                self.rng.normal(self.config.cols_mu, self.config.cols_sigma),
                lo, hi
            ))
            return rows, cols

        else:
            raise ValueError(f"Mode de sampling inconnu: {mode}")


class PositionSampler:
    """
    Génère des positions aléatoires sur une grille (rows x cols).
    Supporte plusieurs modes de distribution.
    """

    def __init__(self, rows: int, cols: int, rng: np.random.Generator):
        self.rows = rows
        self.cols = cols
        self.rng = rng

    def sample(
        self,
        mode: PositionSamplingMode,
        exclude: Optional[List[Position]] = None,
    ) -> Position:
        """Tire une position selon le mode donné."""
        excluded = set(exclude or [])
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in excluded
        ]
        if not candidates:
            raise ValueError("Aucune position disponible pour le sampling.")

        if mode == "uniform":
            idx = self.rng.integers(0, len(candidates))
            return candidates[idx]

        elif mode == "corner_biased":
            # Pondère les positions proches des coins (utile pour les sorties)
            corners = [
                (0, 0), (0, self.cols-1),
                (self.rows-1, 0), (self.rows-1, self.cols-1)
            ]
            weights = np.array([
                1.0 / (1 + min(
                    abs(r - cr) + abs(c - cc)
                    for cr, cc in corners
                ))
                for r, c in candidates
            ], dtype=float)
            weights /= weights.sum()
            idx = self.rng.choice(len(candidates), p=weights)
            return candidates[idx]

        else:
            raise ValueError(f"Mode de position inconnu: {mode}")

    def sample_n_distinct(
        self,
        n: int,
        mode: PositionSamplingMode,
        exclude: Optional[List[Position]] = None,
    ) -> List[Position]:
        """Tire n positions distinctes."""
        excluded = list(exclude or [])
        positions = []
        for _ in range(n):
            pos = self.sample(mode, exclude=excluded + positions)
            positions.append(pos)
        return positions


def spawn_configuration(
    grid_config: GridConfig,
    agent_config: AgentConfig,
    rng: np.random.Generator,
) -> Tuple[int, int, Position, Position, List[Position]]:
    """
    Génère une configuration complète :
    rows, cols, mario_start, monster_start, exit_positions.

    Garantit que Mario et le Monstre ne commencent pas au même endroit
    et respectent la distance minimale configurée.
    """
    from src.environment.grid import Grid

    grid_sampler = GridSampler(grid_config, rng)
    rows, cols = grid_sampler.sample_size()

    pos_sampler = PositionSampler(rows, cols, rng)

    # --- Sorties ---
    exits = pos_sampler.sample_n_distinct(
        grid_config.num_exits,
        mode=grid_config.exit_sampling_mode,
    )

    # --- Mario ---
    if agent_config.mario_sampling_mode == "fixed" and agent_config.mario_start:
        mario_pos = agent_config.mario_start
    else:
        mario_pos = pos_sampler.sample(
            agent_config.mario_sampling_mode,
            exclude=exits,
        )

    # --- Monstre ---
    min_dist = agent_config.min_initial_distance
    max_attempts = 200

    if agent_config.monster_sampling_mode == "fixed" and agent_config.monster_start:
        monster_pos = agent_config.monster_start
    else:
        for attempt in range(max_attempts):
            candidate = pos_sampler.sample(
                agent_config.monster_sampling_mode,
                exclude=exits + [mario_pos],
            )
            dist = abs(candidate[0] - mario_pos[0]) + abs(candidate[1] - mario_pos[1])
            if dist >= min_dist:
                monster_pos = candidate
                break
        else:
            # Fallback : prendre la position la plus éloignée disponible
            all_pos = [
                (r, c)
                for r in range(rows) for c in range(cols)
                if (r, c) not in exits and (r, c) != mario_pos
            ]
            monster_pos = max(
                all_pos,
                key=lambda p: abs(p[0] - mario_pos[0]) + abs(p[1] - mario_pos[1])
            )

    return rows, cols, mario_pos, monster_pos, exits
