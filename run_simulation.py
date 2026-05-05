"""
run_simulation.py
-----------------
Point d'entrée principal pour lancer les simulations Monte Carlo.

Exemples d'utilisation :
  # Simulation rapide (1000 runs, toutes stratégies)
  python run_simulation.py --runs 1000

  # Simulation complète avec export CSV
  python run_simulation.py --runs 10000 --output data/raw/results.csv

  # Stratégie spécifique uniquement
  python run_simulation.py --mario-strategy astar --monster-strategy aggressive --runs 5000

  # Grille fixe 10x10
  python run_simulation.py --grid-mode fixed --rows 10 --cols 10 --runs 2000

  # Analyse post-simulation
  python run_simulation.py --analyze data/raw/results.csv
"""

import argparse
import os
import sys

# Ajouter le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import (
    ExperimentConfig, GridConfig, AgentConfig,
    SimulationConfig, MonteCarloConfig
)
from src.simulation.monte_carlo import MonteCarloRunner
from src.analysis.stats import load_results, print_report, plot_survival_rates, plot_steps_distribution, plot_heatmap_outcomes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mario Escape Simulation — Monte Carlo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Simulation
    parser.add_argument("--runs", type=int, default=1000,
                        help="Nombre total de simulations (défaut: 1000)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Nombre max de pas par simulation (défaut: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire de base (défaut: 42)")

    # Stratégies
    parser.add_argument("--mario-strategy", type=str, default=None,
                        choices=["random", "greedy", "astar"],
                        help="Forcer une stratégie unique pour Mario")
    parser.add_argument("--monster-strategy", type=str, default=None,
                        choices=["random", "aggressive", "semi_aggressive"],
                        help="Forcer une stratégie unique pour le Monstre")

    # Grille
    parser.add_argument("--grid-mode", type=str, default="uniform",
                        choices=["fixed", "uniform", "normal"],
                        help="Mode de sampling de la grille (défaut: uniform)")
    parser.add_argument("--rows", type=int, default=8,
                        help="Lignes (mode fixed uniquement)")
    parser.add_argument("--cols", type=int, default=8,
                        help="Colonnes (mode fixed uniquement)")
    parser.add_argument("--exits", type=int, default=2,
                        help="Nombre de sorties (défaut: 2)")

    # Output
    parser.add_argument("--output", type=str, default="data/raw/simulation_results.csv",
                        help="Chemin du fichier CSV de sortie")
    parser.add_argument("--quiet", action="store_true",
                        help="Désactiver la barre de progression")

    # Analyse uniquement
    parser.add_argument("--analyze", type=str, default=None, metavar="CSV_PATH",
                        help="Analyser un CSV existant sans lancer de simulation")
    parser.add_argument("--plots-dir", type=str, default="data/processed",
                        help="Dossier de sauvegarde des graphiques")

    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    # Stratégies
    mario_strategies = (
        [args.mario_strategy] if args.mario_strategy
        else ["random", "greedy", "astar"]
    )
    monster_strategies = (
        [args.monster_strategy] if args.monster_strategy
        else ["random", "aggressive"]
    )

    grid_config = GridConfig(
        sampling_mode=args.grid_mode,
        rows=args.rows,
        cols=args.cols,
        num_exits=args.exits,
    )

    mc_config = MonteCarloConfig(
        num_runs=args.runs,
        output_path=args.output,
        mario_strategies=mario_strategies,
        monster_strategies=monster_strategies,
        cross_strategies=True,
        verbose=not args.quiet,
        base_seed=args.seed,
    )

    sim_config = SimulationConfig(max_steps=args.max_steps)

    return ExperimentConfig(
        grid=grid_config,
        simulation=sim_config,
        monte_carlo=mc_config,
    )


def run_analysis(csv_path: str, plots_dir: str):
    print(f"\n📊 Chargement : {csv_path}")
    df = load_results(csv_path)
    print_report(df)

    os.makedirs(plots_dir, exist_ok=True)

    plot_survival_rates(df, save_path=os.path.join(plots_dir, "survival_rates.png"))
    plot_steps_distribution(df, save_path=os.path.join(plots_dir, "steps_distribution.png"))
    plot_heatmap_outcomes(df, save_path=os.path.join(plots_dir, "heatmap_outcomes.png"))

    print(f"\n📈 Graphiques sauvegardés dans : {plots_dir}/")


def main():
    args = parse_args()

    if args.analyze:
        run_analysis(args.analyze, args.plots_dir)
        return

    config = build_config(args)

    print(f"\n🎮 Mario Escape Simulation — Monte Carlo")
    print(f"   Runs         : {config.monte_carlo.num_runs}")
    print(f"   Mario strats : {config.monte_carlo.mario_strategies}")
    print(f"   Monster strats: {config.monte_carlo.monster_strategies}")
    print(f"   Grid mode    : {config.grid.sampling_mode}")
    print(f"   Output       : {config.monte_carlo.output_path}")

    runner = MonteCarloRunner(config)
    results = runner.run()

    # Analyse automatique après simulation
    if os.path.exists(config.monte_carlo.output_path):
        run_analysis(config.monte_carlo.output_path, args.plots_dir)


if __name__ == "__main__":
    main()
