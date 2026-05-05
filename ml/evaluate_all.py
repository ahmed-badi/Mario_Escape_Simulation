"""
ml/evaluate_all.py
-------------------
Module d'évaluation comparatif : compare toutes les stratégies
Mario sur le même ensemble d'environnements.

Stratégies comparées :
  1. Random      (baseline aléatoire)
  2. Greedy      (heuristique Manhattan)
  3. A*          (pathfinding BFS avec évitement)
  4. DQN Agent   (agent RL entraîné)

Produit :
  - Tableau comparatif dans la console
  - data/processed/strategy_comparison.png
  - data/processed/strategy_comparison.csv

Usage :
  python -m ml.evaluate_all
  python -m ml.evaluate_all --episodes 500 --monster aggressive
  python -m ml.evaluate_all --dqn-model data/models/dqn_best.pt
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm

from ml.rl.environment_wrapper import MarioEscapeEnv, EnvConfig, ACTION_DELTAS
from src.utils.config import GridConfig, AgentConfig
from src.environment.grid import Grid


# ---------------------------------------------------------------------------
# Wrappers de stratégies classiques dans l'interface Env
# ---------------------------------------------------------------------------

class ClassicStrategyWrapper:
    """
    Adapte les stratégies existantes (random, greedy, astar)
    pour fonctionner avec MarioEscapeEnv.
    """

    def __init__(self, strategy_name: str, rng: np.random.Generator):
        from src.strategies.mario_strategies import get_mario_strategy
        self.strategy = get_mario_strategy(strategy_name, rng=rng)

    def select_action(self, obs: np.ndarray, env: MarioEscapeEnv) -> int:
        """Appelle la stratégie originale et traduit en action entière."""
        next_pos = self.strategy.next_move(
            grid=env.grid,
            mario_pos=env.mario_pos,
            monster_pos=env.monster.position,
        )
        # Trouver quelle action correspond à ce déplacement
        dr = next_pos[0] - env.mario_pos[0]
        dc = next_pos[1] - env.mario_pos[1]
        for action, (adr, adc) in ACTION_DELTAS.items():
            if adr == dr and adc == dc:
                return action
        return 0  # fallback


class DQNStrategyWrapper:
    """Wrapper autour de l'agent DQN pour l'évaluation."""

    def __init__(self, agent):
        self.agent = agent

    def select_action(self, obs: np.ndarray, env: MarioEscapeEnv) -> int:
        return self.agent.select_action(obs, eval_mode=True)


# ---------------------------------------------------------------------------
# Fonction d'évaluation générique
# ---------------------------------------------------------------------------

def evaluate_strategy(
    strategy_name: str,
    wrapper,
    env_config: EnvConfig,
    n_episodes: int = 300,
    seed: int = 2024,
) -> Dict:
    """
    Évalue une stratégie sur n_episodes épisodes.
    Retourne un dict de statistiques.
    """
    env = MarioEscapeEnv(config=env_config, seed=seed)
    outcomes = []
    rewards = []
    steps_list = []

    for _ in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0

        while True:
            action = wrapper.select_action(obs, env)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                outcomes.append(info["outcome"])
                steps_list.append(info["step"])
                rewards.append(ep_reward)
                break

    total = len(outcomes)
    return {
        "strategy": strategy_name,
        "escape": outcomes.count("escape"),
        "caught": outcomes.count("caught"),
        "timeout": outcomes.count("timeout"),
        "escape_rate": outcomes.count("escape") / total,
        "caught_rate": outcomes.count("caught") / total,
        "timeout_rate": outcomes.count("timeout") / total,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "n_episodes": total,
    }


# ---------------------------------------------------------------------------
# Comparaison complète
# ---------------------------------------------------------------------------

def compare_all(
    n_episodes: int = 300,
    monster_strategy: str = "aggressive",
    dqn_model_path: str = "data/models/dqn_best.pt",
    output_dir: str = "data/processed",
    models_dir: str = "data/models",
):
    os.makedirs(output_dir, exist_ok=True)

    env_config = EnvConfig(
        grid_config=GridConfig(sampling_mode="uniform", num_exits=2),
        agent_config=AgentConfig(min_initial_distance=2),
        monster_strategy=monster_strategy,
        max_steps=200,
    )

    rng = np.random.default_rng(2024)
    strategies = {}

    # --- Stratégies classiques ---
    for name in ["random", "greedy", "astar"]:
        strategies[name] = ClassicStrategyWrapper(name, rng=np.random.default_rng(2024))

    # --- Agent DQN ---
    dqn_loaded = False
    if os.path.exists(dqn_model_path):
        try:
            from ml.rl.dqn_agent import DQNAgent
            agent = DQNAgent.load(dqn_model_path)
            strategies["DQN Agent"] = DQNStrategyWrapper(agent)
            dqn_loaded = True
        except Exception as e:
            print(f"  ⚠️  Impossible de charger DQN : {e}")
    else:
        print(f"  ℹ️  Modèle DQN non trouvé ({dqn_model_path}) — évaluation sans DQN.")

    # --- Évaluation ---
    print(f"\n{'='*60}")
    print(f"  COMPARAISON DES STRATÉGIES ({n_episodes} épisodes chacune)")
    print(f"  Monstre : {monster_strategy}")
    print(f"{'='*60}")

    all_results = []
    for name, wrapper in strategies.items():
        print(f"  Évaluation : {name}...", end="", flush=True)
        result = evaluate_strategy(name, wrapper, env_config, n_episodes)
        all_results.append(result)
        print(f"  escape={result['escape_rate']:.2%}  caught={result['caught_rate']:.2%}  "
              f"timeout={result['timeout_rate']:.2%}  reward={result['mean_reward']:+.3f}")

    # --- Tableau ---
    df = pd.DataFrame(all_results)
    df = df.sort_values("escape_rate", ascending=False)

    print(f"\n{'='*60}")
    print(f"  CLASSEMENT FINAL")
    print(f"{'='*60}")
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        medal = ["🥇", "🥈", "🥉", "  "][min(rank-1, 3)]
        print(
            f"  {medal} {rank}. {row['strategy']:<15}"
            f"  escape={row['escape_rate']:.2%}"
            f"  caught={row['caught_rate']:.2%}"
            f"  steps={row['mean_steps']:.1f}"
        )

    csv_path = os.path.join(output_dir, "strategy_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 CSV → {csv_path}")

    # --- Graphique ---
    _plot_comparison(df, monster_strategy, output_dir, dqn_loaded)

    return df


def _plot_comparison(df: pd.DataFrame, monster: str, output_dir: str, dqn_loaded: bool):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    colors_map = {
        "random": "#95a5a6",
        "greedy": "#3498db",
        "astar": "#2ecc71",
        "DQN Agent": "#e74c3c",
    }
    strategies = df["strategy"].tolist()
    colors = [colors_map.get(s, "#f39c12") for s in strategies]

    # 1 — Escape rate
    ax = axes[0]
    bars = ax.bar(strategies, df["escape_rate"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Taux d'escape", fontweight="bold")
    ax.set_ylabel("Escape rate")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, df["escape_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 2 — Répartition des outcomes (stacked bar)
    ax = axes[1]
    x = np.arange(len(strategies))
    w = 0.6
    ax.bar(x, df["escape_rate"],  w, label="Escape",  color="#2ecc71", alpha=0.85)
    ax.bar(x, df["caught_rate"],  w, label="Caught",  color="#e74c3c", alpha=0.85,
           bottom=df["escape_rate"])
    ax.bar(x, df["timeout_rate"], w, label="Timeout", color="#95a5a6", alpha=0.85,
           bottom=df["escape_rate"] + df["caught_rate"])
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=10)
    ax.set_title("Répartition des outcomes", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 3 — Reward moyen
    ax = axes[2]
    bars = ax.bar(strategies, df["mean_reward"], color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Reward moyen par épisode", fontweight="bold")
    ax.set_ylabel("Reward")
    for bar, val in zip(bars, df["mean_reward"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.01 if val >= 0 else -0.05),
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Comparaison des stratégies Mario\n(monstre: {monster}, "
        f"{'DQN inclus' if dqn_loaded else 'DQN non disponible'})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "strategy_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📈 Graphique → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparer toutes les stratégies Mario")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--monster", default="aggressive",
                        choices=["random", "aggressive", "semi_aggressive"])
    parser.add_argument("--dqn-model", default="data/models/dqn_best.pt")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    compare_all(
        n_episodes=args.episodes,
        monster_strategy=args.monster,
        dqn_model_path=args.dqn_model,
        output_dir=args.output_dir,
    )