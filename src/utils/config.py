"""
src/utils/config.py
-------------------
Dataclasses de configuration pour la simulation.
Toutes les options sont centralisées ici pour faciliter
la sérialisation YAML et la reproductibilité des expériences.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple


GridSamplingMode = Literal["fixed", "uniform", "normal"]
PositionSamplingMode = Literal["fixed", "uniform", "corner_biased"]
StrategyName = Literal["random", "greedy", "astar"]
MonsterStrategyName = Literal["random", "aggressive", "semi_aggressive"]


@dataclass
class GridConfig:
    """Configuration de la grille 2D."""
    sampling_mode: GridSamplingMode = "uniform"

    # Mode "fixed"
    rows: int = 8
    cols: int = 8

    # Mode "uniform" : tirage U[min, max]
    rows_min: int = 5
    rows_max: int = 15
    cols_min: int = 5
    cols_max: int = 15

    # Mode "normal" : tirage N(mu, sigma) tronqué
    rows_mu: float = 8.0
    rows_sigma: float = 2.0
    cols_mu: float = 8.0
    cols_sigma: float = 2.0
    size_min: int = 4
    size_max: int = 20

    # Nombre de sorties
    num_exits: int = 2
    exit_sampling_mode: PositionSamplingMode = "uniform"


@dataclass
class AgentConfig:
    """Configuration du placement des agents."""
    mario_sampling_mode: PositionSamplingMode = "uniform"
    monster_sampling_mode: PositionSamplingMode = "uniform"

    # Positions fixes (utilisées si mode == "fixed")
    mario_start: Optional[Tuple[int, int]] = None
    monster_start: Optional[Tuple[int, int]] = None

    # Distance minimale initiale Mario <-> Monstre
    min_initial_distance: int = 3


@dataclass
class SimulationConfig:
    """Configuration d'une simulation individuelle."""
    max_steps: int = 200
    mario_strategy: StrategyName = "greedy"
    monster_strategy: MonsterStrategyName = "aggressive"
    seed: Optional[int] = None


@dataclass
class MonteCarloConfig:
    """Configuration du runner Monte Carlo."""
    num_runs: int = 10_000
    output_path: str = "data/raw/simulation_results.csv"
    mario_strategies: List[StrategyName] = field(
        default_factory=lambda: ["random", "greedy", "astar"]
    )
    monster_strategies: List[MonsterStrategyName] = field(
        default_factory=lambda: ["random", "aggressive"]
    )
    # Si True, toutes les combinaisons de stratégies sont testées
    cross_strategies: bool = True
    # Nombre de workers parallèles (1 = séquentiel)
    n_jobs: int = 1
    verbose: bool = True
    base_seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration complète d'une expérience."""
    name: str = "default_experiment"
    grid: GridConfig = field(default_factory=GridConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
