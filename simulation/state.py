from typing import Dict, Any, List


def init_state() -> Dict[str, Any]:
    return {
        "running": False,
        "paused": False,
        "mode": "single",
        "mario_strategy": "random",
        "monster_strategy": "aggressive",
        "grid_size": 10,
        "num_episodes": 30,
        "speed_ms": 400,
        "show_path": True,
        "fast_forward": False,
        "status": "ready",
        "episode": 0,
        "step": 0,
        "outcome": "ready",
        "last_reward": 0.0,
        "episode_history": [],
        "outcome_counts": {"escape": 0, "caught": 0, "timeout": 0},
        "total_reward": 0.0,
        "total_steps": 0,
        "current_combo": "",
        "combo_index": 0,
        "combos": [],
        "benchmark_results": [],
        "grid": {
            "rows": 10,
            "cols": 10,
            "exits": [],
            "mario": [0, 0],
            "monster": [0, 0],
            "path": [],
        },
        "best_combo": "",
        "worst_combo": "",
        "seed": 0,
    }


def build_benchmark_combos(dqn_available: bool) -> List[Dict[str, str]]:
    mario_options = ["random", "greedy", "astar"]
    if dqn_available:
        mario_options.append("dqn")
    monster_options = ["random", "aggressive", "semi_aggressive"]
    return [
        {"mario": m, "monster": mo, "label": f"{m.upper()} vs {mo.upper()}"}
        for m in mario_options
        for mo in monster_options
    ]


def compute_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    episodes = max(state["episode"], 1)
    escape = state["outcome_counts"]["escape"]
    caught = state["outcome_counts"]["caught"]
    timeout = state["outcome_counts"]["timeout"]
    return {
        "escape_rate": round(100.0 * escape / episodes, 1),
        "caught_rate": round(100.0 * caught / episodes, 1),
        "timeout_rate": round(100.0 * timeout / episodes, 1),
        "avg_reward": round(state["total_reward"] / episodes, 3),
        "avg_steps": round(state["total_steps"] / episodes, 1),
    }
