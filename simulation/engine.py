import json
import os
from typing import Dict, Any, Optional

from ml.rl.dqn_agent import DQNAgent
from ml.rl.environment_wrapper import MarioEscapeEnv
from src.utils.config import GridConfig
from src.strategies.mario_strategies import get_mario_strategy
from src.strategies.monster_strategies import get_monster_strategy


env: Optional[MarioEscapeEnv] = None
agent: Optional[DQNAgent] = None


def init_runtime(grid_size: int = 10, seed: int = 0) -> None:
    global env
    config = GridConfig(row_count=grid_size, col_count=grid_size)
    env = MarioEscapeEnv(config=config, seed=seed)


def load_dqn_model(model_path: str) -> DQNAgent:
    global agent
    if agent is not None:
        return agent
    agent = DQNAgent.load(model_path)
    return agent


def is_dqn_available() -> bool:
    model_path = os.path.join("data", "models", "dqn_agent.pt")
    return os.path.exists(model_path)


def step_simulation(state: Dict[str, Any]) -> Dict[str, Any]:
    global env, agent
    if env is None:
        init_runtime(grid_size=state.get("grid_size", 10), seed=state.get("seed", 0))
    if state["mode"] == "benchmark":
        return state

    env.reset()
    mario = get_mario_strategy(state["mario_strategy"], env, agent)
    monster = get_monster_strategy(state["monster_strategy"], env)

    done = False
    reward = 0.0
    steps = 0

    while not done:
        action = mario.select_action(env.state)
        _, step_reward, done, _ = env.step(action)
        reward += step_reward
        steps += 1
        if done:
            break
        monster_action = monster.select_action(env.state)
        _, _, done, _ = env.step(monster_action)
        steps += 1

    state["episode"] += 1
    state["step"] = steps
    state["last_reward"] = reward
    state["total_steps"] += steps
    state["total_reward"] += reward

    if env.last_event == "escape":
        outcome = "escape"
    elif env.last_event == "caught":
        outcome = "caught"
    else:
        outcome = "timeout"

    state["outcome"] = outcome
    state["outcome_counts"][outcome] += 1
    state["status"] = "completed"
    state["running"] = False
    state["paused"] = False

    state["grid"] = {
        "rows": env.config.row_count,
        "cols": env.config.col_count,
        "exits": env.exits,
        "mario": env.mario_position,
        "monster": env.monster_position,
        "path": env.path if hasattr(env, "path") else [],
    }

    return state
