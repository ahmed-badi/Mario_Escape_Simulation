import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from dash.dash_table import DataTable

# Ensure imports work from project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.rl.environment_wrapper import EnvConfig, MarioEscapeEnv
from ml.rl.dqn_agent import DQNAgent
from src.strategies.mario_strategies import get_mario_strategy
from src.utils.config import AgentConfig, GridConfig

ACTION_DELTAS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}

DQN_PATHS = [
    os.path.join(ROOT_DIR, "data", "models", "dqn_best.pt"),
    os.path.join(ROOT_DIR, "data", "models", "dqn_agent.pt"),
]

SIM_CONTEXT: Dict[str, Any] = {}
DQN_AGENT: Optional[DQNAgent] = None


def load_dqn_model(silent: bool = False) -> Optional[DQNAgent]:
    global DQN_AGENT
    if DQN_AGENT is not None:
        return DQN_AGENT

    for path in DQN_PATHS:
        if os.path.exists(path):
            DQN_AGENT = DQNAgent.load(path, device="cpu", silent=silent)
            return DQN_AGENT

    return None


def position_to_action(current: Tuple[int, int], target: Tuple[int, int]) -> int:
    for action, (dr, dc) in ACTION_DELTAS.items():
        if (current[0] + dr, current[1] + dc) == target:
            return action
    return 0


def build_env_config(grid_size: int, monster_strategy: str, max_steps: int = 200) -> EnvConfig:
    grid_config = GridConfig(
        sampling_mode="fixed",
        rows=grid_size,
        cols=grid_size,
        num_exits=2,
    )
    agent_config = AgentConfig()
    return EnvConfig(
        grid_config=grid_config,
        agent_config=agent_config,
        monster_strategy=monster_strategy,
        max_steps=max_steps,
    )


def default_state() -> Dict[str, Any]:
    return {
        "running": False,
        "mode": "single",
        "mario_strategy": "random",
        "monster_strategy": "aggressive",
        "grid_size": 12,
        "num_episodes": 100,
        "speed_ms": 300,
        "show_path": True,
        "fast_forward": False,
        "status": "ready",
        "current_episode": 0,
        "current_step": 0,
        "current_outcome": "ready",
        "last_reward": 0.0,
        "episode_history": [],
        "outcome_counts": {"escape": 0, "caught": 0, "timeout": 0},
        "total_reward": 0.0,
        "total_steps": 0,
        "current_combo": "",
        "combo_index": 0,
        "combos": [],
        "benchmark_results": [],
        "grid_data": {
            "rows": 10,
            "cols": 10,
            "exits": [],
            "mario": [0, 0],
            "monster": [0, 0],
            "path": [],
        },
        "best_combo": "",
        "worst_combo": "",
        "initialized": False,
        "run_id": 0,
    }


def build_benchmark_combos() -> List[Dict[str, str]]:
    dqn_available = load_dqn_model(silent=True) is not None
    mario_options = ["random", "greedy", "astar"]
    if dqn_available:
        mario_options.append("dqn")
    monster_options = ["random", "aggressive", "semi_aggressive"]
    return [
        {"mario": m, "monster": mo, "label": f"{m.upper()} vs {mo.upper()}"}
        for m in mario_options
        for mo in monster_options
    ]


def get_grid_path(env: MarioEscapeEnv, show_path: bool) -> List[List[int]]:
    if not show_path:
        return []
    nearest_exit, _ = env.grid.nearest_exit(env.mario_pos)
    if nearest_exit is None:
        return []
    path = env.grid.shortest_path(env.mario_pos, nearest_exit)
    return [[int(r), int(c)] for r, c in path]


def make_grid_snapshot(env: MarioEscapeEnv, show_path: bool) -> Dict[str, Any]:
    return {
        "rows": env.grid.rows,
        "cols": env.grid.cols,
        "exits": [[r, c] for r, c in env.grid.exits],
        "mario": [int(env.mario_pos[0]), int(env.mario_pos[1])],
        "monster": [int(env.monster.position[0]), int(env.monster.position[1])],
        "path": get_grid_path(env, show_path),
    }


def initialize_simulation(data: Dict[str, Any]) -> Dict[str, Any]:
    SIM_CONTEXT.clear()
    if data["mode"] == "benchmark":
        combos = build_benchmark_combos()
        selected_mario = data.get("mario_strategy")
        selected_monster = data.get("monster_strategy")
        if selected_mario and selected_monster:
            matching_combo = next(
                (
                    combo
                    for combo in combos
                    if combo["mario"] == selected_mario and combo["monster"] == selected_monster
                ),
                None,
            )
            if matching_combo:
                combos.remove(matching_combo)
                combos.insert(0, matching_combo)
        data["combos"] = combos
        data["combo_index"] = 0
        data["benchmark_results"] = []
        data["current_combo"] = data["combos"][0]["label"] if data["combos"] else ""
    else:
        data["combos"] = []
        data["combo_index"] = 0
        data["benchmark_results"] = []
        data["current_combo"] = f"{data['mario_strategy'].upper()} vs {data['monster_strategy'].upper()}"

    data["episode_history"] = []
    data["outcome_counts"] = {"escape": 0, "caught": 0, "timeout": 0}
    data["total_reward"] = 0.0
    data["total_steps"] = 0
    data["current_episode"] = 0
    data["current_step"] = 0
    data["current_outcome"] = "ready"
    data["best_combo"] = ""
    data["worst_combo"] = ""
    data["initialized"] = False
    data["run_id"] = 1
    data["status"] = "ready"

    build_next_episode(data)
    return data


def build_next_episode(data: Dict[str, Any]) -> None:
    if data["mode"] == "benchmark":
        if data["combo_index"] >= len(data["combos"]):
            data["status"] = "finished"
            data["running"] = False
            return
        combo = data["combos"][data["combo_index"]]
        mario_strategy = combo["mario"]
        monster_strategy = combo["monster"]
        data["current_combo"] = combo["label"]
    else:
        mario_strategy = data["mario_strategy"]
        monster_strategy = data["monster_strategy"]

    env_config = build_env_config(data["grid_size"], monster_strategy)
    env = MarioEscapeEnv(config=env_config, seed=np.random.randint(0, 1000000))

    if mario_strategy == "dqn":
        dqn_agent = load_dqn_model()
        if dqn_agent is None:
            data["status"] = "error: missing dqn model"
            data["running"] = False
            return
        SIM_CONTEXT["dqn_agent"] = dqn_agent
        SIM_CONTEXT["mario_strategy_obj"] = None
    else:
        SIM_CONTEXT["dqn_agent"] = None
        SIM_CONTEXT["mario_strategy_obj"] = get_mario_strategy(mario_strategy, rng=np.random.default_rng())

    SIM_CONTEXT["env"] = env
    SIM_CONTEXT["grid_size"] = data["grid_size"]
    SIM_CONTEXT["show_path"] = data["show_path"]
    data["grid_data"] = {
        "rows": env.grid.rows,
        "cols": env.grid.cols,
        "exits": [[r, c] for r, c in env.grid.exits],
        "mario": [int(env.mario_pos[0]), int(env.mario_pos[1])],
        "monster": [int(env.monster.position[0]), int(env.monster.position[1])],
        "path": get_grid_path(env, data["show_path"]),
    }
    data["current_step"] = 0
    data["current_outcome"] = "running"
    data["status"] = "running"
    data["initialized"] = True


def determine_action(data: Dict[str, Any]) -> int:
    env: MarioEscapeEnv = SIM_CONTEXT["env"]
    if data["mario_strategy"] == "dqn" or data["mode"] == "benchmark" and data["combos"] and data["combos"][data["combo_index"]]["mario"] == "dqn":
        dqn_agent = SIM_CONTEXT.get("dqn_agent") or load_dqn_model()
        if dqn_agent is None:
            raise RuntimeError("DQN model is not available")
        return dqn_agent.select_action(env._get_obs(), eval_mode=True)

    strategy = SIM_CONTEXT.get("mario_strategy_obj")
    if strategy is None:
        strategy_name = data["mario_strategy"]
        strategy = get_mario_strategy(strategy_name, rng=np.random.default_rng())
        SIM_CONTEXT["mario_strategy_obj"] = strategy

    next_position = strategy.next_move(env.grid, env.mario_pos, env.monster.position)
    return position_to_action(env.mario_pos, next_position)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    sorted_results = sorted(results, key=lambda item: item["escape_rate"], reverse=True)
    best = sorted_results[0]["label"] if sorted_results else ""
    worst = sorted_results[-1]["label"] if sorted_results else ""
    return {"best_combo": best, "worst_combo": worst}


def update_episode_metrics(data: Dict[str, Any], info: Dict[str, Any], reward: float) -> None:
    data["total_reward"] += reward
    data["total_steps"] += info["step"]
    if info["outcome"] in data["outcome_counts"]:
        data["outcome_counts"][info["outcome"]] += 1
    data["episode_history"].append(
        {
            "episode": data["current_episode"],
            "reward": round(data["total_reward"] / max(data["current_episode"], 1), 3),
            "steps": info["step"],
            "outcome": info["outcome"],
            "combo": data["current_combo"],
        }
    )


def compute_aggregate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    episodes = max(data["current_episode"], 1)
    escape = data["outcome_counts"]["escape"]
    caught = data["outcome_counts"]["caught"]
    timeout = data["outcome_counts"]["timeout"]
    return {
        "escape_rate": round(100.0 * escape / episodes, 1),
        "caught_rate": round(100.0 * caught / episodes, 1),
        "timeout_rate": round(100.0 * timeout / episodes, 1),
        "avg_reward": round(data["total_reward"] / episodes, 3),
        "avg_steps": round(data["total_steps"] / episodes, 1),
    }


def build_grid_figure(grid: Dict[str, Any]) -> go.Figure:
    rows, cols = grid["rows"], grid["cols"]
    z = np.zeros((rows, cols), dtype=int)
    for r, c in grid["exits"]:
        z[r, c] = 1
    mr, mc = grid["mario"]
    z[mr, mc] = 2
    br, bc = grid["monster"]
    z[br, bc] = 3

    colors = ["#0d1117", "#fbc531", "#1e90ff", "#ff4757"]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorscale=[
                [0.0, colors[0]],
                [0.25, colors[1]],
                [0.5, colors[2]],
                [0.75, colors[3]],
                [1.0, colors[3]],
            ],
            showscale=False,
            hoverinfo="none",
            xgap=1,
            ygap=1,
        )
    )
    path = grid["path"]
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                marker=dict(size=10, color="#ffffff", line=dict(color="#1e90ff", width=2)),
                line=dict(color="#1e90ff", width=3),
                name="Path",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[mc],
            y=[mr],
            mode="markers",
            marker=dict(size=18, color="#1e90ff", symbol="diamond"),
            name="Mario",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[bc],
            y=[br],
            mode="markers",
            marker=dict(size=18, color="#ff4757", symbol="x"),
            name="Monster",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_traces(hoverinfo="skip")
    return fig


def build_line_chart(x: List[int], y: List[float], title: str, ytitle: str) -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#00cc96", width=3),
            marker=dict(size=8, color="#00cc96", line=dict(width=1, color="#ffffff")),
            fill="tozeroy",
            fillcolor="rgba(0, 204, 150, 0.2)",
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title=ytitle,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#222222", zeroline=False),
    )
    return fig


def build_outcome_chart(data: Dict[str, Any]) -> go.Figure:
    counts = data["outcome_counts"]
    labels = ["Escape", "Caught", "Timeout"]
    values = [counts[k.lower()] for k in labels]
    fig = go.Figure(
        data=go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=["#2ecc71", "#e74c3c", "#95a5a6"]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title="Outcome distribution",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    return fig


def build_benchmark_bar(results: List[Dict[str, Any]]) -> go.Figure:
    labels = [item["label"] for item in results]
    escape_rates = [item["escape_rate"] for item in results]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=escape_rates,
                marker_color="#4dabf7",
                text=[f"{v:.1f}%" for v in escape_rates],
                textposition="auto",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Benchmark escape rate by combo",
        xaxis_tickangle=-45,
        yaxis_title="Escape rate (%)",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=20, t=50, b=160),
        height=420,
    )
    return fig


def build_summary_cards(metrics: Dict[str, Any]) -> List[dbc.Card]:
    return [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6(label, className="card-title text-muted"),
                    html.H4(value, className="card-text"),
                ]
            ),
            className="mb-3 shadow-sm",
            color="dark",
            inverse=True,
        )
        for label, value in [
            ("Escape rate", f"{metrics['escape_rate']}%"),
            ("Caught rate", f"{metrics['caught_rate']}%"),
            ("Timeout rate", f"{metrics['timeout_rate']}%"),
            ("Avg reward", f"{metrics['avg_reward']}"),
            ("Avg steps", f"{metrics['avg_steps']}"),
        ]
    ]


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Mario Escape Dashboard"

app.layout = dbc.Container(
    [
        html.H1("Mario Escape Simulation Dashboard", className="my-4"),
        dcc.Store(id="store-sim-data", data=default_state()),
        dcc.Interval(id="interval", interval=400, n_intervals=0, disabled=True),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Controls", className="card-title"),
                                html.Label("Mode"),
                                dbc.RadioItems(
                                    id="mode-radio",
                                    options=[
                                        {"label": "Single run", "value": "single"},
                                        {"label": "Benchmark", "value": "benchmark"},
                                    ],
                                    value="single",
                                    inline=True,
                                ),
                                html.Hr(),
                                html.Label("Mario strategy"),
                                dcc.Dropdown(
                                    id="mario-strategy",
                                    options=[
                                        {"label": "Random", "value": "random"},
                                        {"label": "Greedy", "value": "greedy"},
                                        {"label": "A*", "value": "astar"},
                                        {"label": "DQN", "value": "dqn"},
                                    ],
                                    value="random",
                                ),
                                html.Br(),
                                html.Label("Monster strategy"),
                                dcc.Dropdown(
                                    id="monster-strategy",
                                    options=[
                                        {"label": "Random", "value": "random"},
                                        {"label": "Aggressive", "value": "aggressive"},
                                        {"label": "Semi-aggressive", "value": "semi_aggressive"},
                                    ],
                                    value="aggressive",
                                ),
                                html.Br(),
                                html.Label("Grid size"),
                                dcc.Slider(
                                    id="grid-size",
                                    min=5,
                                    max=20,
                                    step=1,
                                    value=12,
                                    marks={i: str(i) for i in [5, 10, 15, 20]},
                                ),
                                html.Br(),
                                html.Label("Episodes"),
                                dcc.Slider(
                                    id="num-episodes",
                                    min=5,
                                    max=500,
                                    step=5,
                                    value=100,
                                    marks={i: str(i) for i in [10, 50, 150, 300, 500]},
                                ),
                                html.Br(),
                                html.Label("Simulation speed (ms)"),
                                dcc.Slider(
                                    id="speed-ms",
                                    min=50,
                                    max=1000,
                                    step=50,
                                    value=300,
                                    marks={i: str(i) for i in [50, 200, 400, 600, 800, 1000]},
                                ),
                                html.Br(),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Show path trail", "value": "show_path"},
                                        {"label": "Fast-forward", "value": "fast_forward"},
                                    ],
                                    value=["show_path"],
                                    id="extra-options",
                                    inline=False,
                                ),
                                html.Br(),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Start", id="btn-start", color="success", n_clicks=0),
                                        dbc.Button("Pause", id="btn-pause", color="warning", n_clicks=0),
                                        dbc.Button("Reset", id="btn-reset", color="secondary", n_clicks=0),
                                    ],
                                    size="lg",
                                ),
                                html.Div(id="control-hint", className="mt-3 text-muted"),
                            ]
                        ),
                        className="mb-4 shadow-sm",
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Live grid simulation", className="card-title"),
                                dcc.Graph(id="grid-figure", config={"displayModeBar": False}),
                                html.Div(
                                    [
                                        html.Span("Status: ", className="fw-bold"),
                                        html.Span(id="status-text"),
                                        html.Br(),
                                        html.Span("Episode: ", className="fw-bold"),
                                        html.Span(id="episode-text"),
                                        html.Br(),
                                        html.Span("Step: ", className="fw-bold"),
                                        html.Span(id="step-text"),
                                        html.Br(),
                                        html.Span("Combo: ", className="fw-bold"),
                                        html.Span(id="combo-text"),
                                    ],
                                    className="small mt-2",
                                ),
                            ]
                        ),
                        className="mb-4 shadow-sm",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Real-time metrics", className="card-title"),
                                html.Div(id="metrics-cards"),
                            ]
                        ),
                        className="mb-4 shadow-sm",
                    ),
                    width=3,
                ),
            ],
            className="g-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="escape-rate-chart"), width=6),
                dbc.Col(dcc.Graph(id="reward-chart"), width=6),
            ],
            className="g-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="outcome-chart"), width=6),
                dbc.Col(dcc.Graph(id="steps-chart"), width=6),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Benchmark results", className="card-title"),
                                DataTable(
                                    id="benchmark-table",
                                    columns=[
                                        {"name": "Combo", "id": "label"},
                                        {"name": "Escape rate", "id": "escape_rate"},
                                        {"name": "Caught rate", "id": "caught_rate"},
                                        {"name": "Timeout rate", "id": "timeout_rate"},
                                        {"name": "Avg reward", "id": "avg_reward"},
                                        {"name": "Avg steps", "id": "avg_steps"},
                                    ],
                                    data=[],
                                    style_header={"backgroundColor": "#1f2c56", "color": "white"},
                                    style_cell={"backgroundColor": "#1a1f2f", "color": "white", "textAlign": "left"},
                                    style_table={"overflowX": "auto"},
                                ),
                                html.Div(id="benchmark-summary", className="mt-3"),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    width=12,
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id="benchmark-bar"), width=12),
            className="g-4",
        ),
        html.Div(id="hidden-debug", style={"display": "none"}),
    ],
    fluid=True,
    className="bg-dark text-white",
)


@app.callback(
    Output("store-sim-data", "data"),
    Output("interval", "disabled"),
    Output("interval", "interval"),
    Output("control-hint", "children"),
    Input("btn-start", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("interval", "n_intervals"),
    Input("mode-radio", "value"),
    State("mario-strategy", "value"),
    State("monster-strategy", "value"),
    State("grid-size", "value"),
    State("num-episodes", "value"),
    State("speed-ms", "value"),
    State("extra-options", "value"),
    State("store-sim-data", "data"),
)
def update_simulation_state(
    start_click,
    pause_click,
    reset_click,
    n_intervals,
    mode,
    mario_strategy,
    monster_strategy,
    grid_size,
    num_episodes,
    speed_ms,
    extra_options,
    store,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    show_path = "show_path" in (extra_options or [])
    fast_forward = "fast_forward" in (extra_options or [])
    interval_delay = speed_ms or 400

    if store is None:
        store = default_state()

    if trigger_id == "btn-reset":
        next_state = default_state()
        next_state.update(
            {
                "mode": mode,
                "mario_strategy": mario_strategy,
                "monster_strategy": monster_strategy,
                "grid_size": grid_size,
                "num_episodes": num_episodes,
                "speed_ms": speed_ms,
                "show_path": show_path,
                "fast_forward": fast_forward,
            }
        )
        next_state = initialize_simulation(next_state)
        hint = "Simulation reset. Press Start to run."
        return next_state, True, interval_delay, hint

    store["mode"] = mode
    store["mario_strategy"] = mario_strategy
    store["monster_strategy"] = monster_strategy
    store["grid_size"] = grid_size
    store["num_episodes"] = num_episodes
    store["speed_ms"] = speed_ms
    store["show_path"] = show_path
    store["fast_forward"] = fast_forward

    if trigger_id == "mode-radio":
        # Mode changed: reset simulation state without starting it
        store["combos"] = []
        store["combo_index"] = 0
        store["benchmark_results"] = []
        store["current_combo"] = ""
        store["episode_history"] = []
        store["outcome_counts"] = {"escape": 0, "caught": 0, "timeout": 0}
        store["total_reward"] = 0.0
        store["total_steps"] = 0
        store["current_episode"] = 0
        store["current_step"] = 0
        store["current_outcome"] = "ready"
        store["best_combo"] = ""
        store["worst_combo"] = ""
        store["initialized"] = False
        store["run_id"] = store.get("run_id", 0) + 1
        store["status"] = "ready"
        store["running"] = False
        hint = f"Mode changed to {mode}. Select strategies and press Start."
        return store, True, interval_delay, hint

    if trigger_id == "btn-start":
        if not store.get("initialized"):
            store = initialize_simulation(store)
        if isinstance(store.get("status"), str) and store["status"].startswith("error"):
            store["running"] = False
            hint = store["status"]
            return store, True, interval_delay, hint
        store["running"] = True
        store["status"] = "running"
        hint = "Running simulation. Pause to stop updates."
        return store, False, interval_delay, hint

    if trigger_id == "btn-pause":
        store["running"] = False
        store["status"] = "paused"
        hint = "Simulation paused. Press Start to resume."
        return store, True, interval_delay, hint

    if trigger_id != "interval":
        raise PreventUpdate

    if not store.get("running"):
        raise PreventUpdate

    if store.get("status") == "finished":
        store["running"] = False
        hint = "Simulation completed."
        return store, True, interval_delay, hint

    try:
        if not store.get("initialized") or SIM_CONTEXT.get("env") is None:
            store = initialize_simulation(store)

        if store["status"].startswith("error"):
            store["running"] = False
            hint = store["status"]
            return store, True, interval_delay, hint

        env: MarioEscapeEnv = SIM_CONTEXT.get("env")
        if env is None:
            raise RuntimeError("Environment not initialized")
        if getattr(env, "done", False):
            build_next_episode(store)
            env = SIM_CONTEXT.get("env")
            if env is None or getattr(env, "done", False):
                raise RuntimeError("Unable to initialize next environment")
        try:
            action = determine_action(store)
        except RuntimeError as exc:
            store["status"] = str(exc)
            store["running"] = False
            hint = store["status"]
            return store, True, interval_delay, hint

        obs, reward, done, info = env.step(action)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        store["status"] = f"error: {type(exc).__name__} {exc}"
        store["running"] = False
        hint = store["status"]
        return store, True, interval_delay, hint
    store["current_step"] = info["step"]
    store["current_outcome"] = info["outcome"]
    store["last_reward"] = round(reward, 3)
    store["grid_data"] = {
        "rows": env.grid.rows,
        "cols": env.grid.cols,
        "exits": [[r, c] for r, c in env.grid.exits],
        "mario": [int(env.mario_pos[0]), int(env.mario_pos[1])],
        "monster": [int(env.monster.position[0]), int(env.monster.position[1])],
        "path": get_grid_path(env, store["show_path"]),
    }

    if done:
        store["current_episode"] += 1
        update_episode_metrics(store, info, reward)
        if store["mode"] == "single":
            if store["current_episode"] >= store["num_episodes"]:
                store["status"] = "finished"
                store["running"] = False
            else:
                build_next_episode(store)
        else:
            combo = store["combos"][store["combo_index"]]
            combo_results = [item for item in store["benchmark_results"] if item["label"] == combo["label"]]
            if not combo_results:
                combo_results.append(
                    {
                        "label": combo["label"],
                        "escape": 0,
                        "caught": 0,
                        "timeout": 0,
                        "cumulative_reward": 0.0,
                        "total_steps": 0,
                        "episodes": 0,
                    }
                )
                store["benchmark_results"].append(combo_results[0])
            summary = combo_results[0]
            summary[info["outcome"]] += 1
            summary["cumulative_reward"] += reward
            summary["total_steps"] += info["step"]
            summary["episodes"] += 1
            if summary["episodes"] >= store["num_episodes"]:
                summary.update(
                    {
                        "escape_rate": round(100.0 * summary["escape"] / summary["episodes"], 1),
                        "caught_rate": round(100.0 * summary["caught"] / summary["episodes"], 1),
                        "timeout_rate": round(100.0 * summary["timeout"] / summary["episodes"], 1),
                        "avg_reward": round(summary["cumulative_reward"] / summary["episodes"], 3),
                        "avg_steps": round(summary["total_steps"] / summary["episodes"], 1),
                    }
                )
                store["combo_index"] += 1
                if store["combo_index"] >= len(store["combos"]):
                    store["status"] = "finished"
                    store["running"] = False
                    summary_metrics = summarize_results(store["benchmark_results"])
                    store["best_combo"] = summary_metrics["best_combo"]
                    store["worst_combo"] = summary_metrics["worst_combo"]
                else:
                    build_next_episode(store)
            else:
                build_next_episode(store)

    env = SIM_CONTEXT.get("env")
    if env is None:
        raise RuntimeError("Environment not initialized")
    if getattr(env, "done", False):
        build_next_episode(store)
        env = SIM_CONTEXT.get("env")
        if env is None or getattr(env, "done", False):
            raise RuntimeError("Unable to initialize next environment")

    if store["fast_forward"]:
        steps = 2
    else:
        steps = 1
    for _ in range(steps - 1):
        if not store["running"] or store["status"] != "running":
            break
        try:
            action = determine_action(store)
        except RuntimeError as exc:
            store["status"] = str(exc)
            store["running"] = False
            break
        obs, reward, done, info = env.step(action)
        store["current_step"] = info["step"]
        store["current_outcome"] = info["outcome"]
        store["last_reward"] = round(reward, 3)
        store["grid_data"] = {
            "rows": env.grid.rows,
            "cols": env.grid.cols,
            "exits": [[r, c] for r, c in env.grid.exits],
            "mario": [int(env.mario_pos[0]), int(env.mario_pos[1])],
            "monster": [int(env.monster.position[0]), int(env.monster.position[1])],
            "path": get_grid_path(env, store["show_path"]),
        }
        if done:
            store["current_episode"] += 1
            update_episode_metrics(store, info, reward)
            if store["mode"] == "single":
                if store["current_episode"] >= store["num_episodes"]:
                    store["status"] = "finished"
                    store["running"] = False
                else:
                    build_next_episode(store)
            else:
                combo = store["combos"][store["combo_index"]]
                combo_results = [item for item in store["benchmark_results"] if item["label"] == combo["label"]]
                if not combo_results:
                    combo_results.append(
                        {
                            "label": combo["label"],
                            "escape": 0,
                            "caught": 0,
                            "timeout": 0,
                            "cumulative_reward": 0.0,
                            "total_steps": 0,
                            "episodes": 0,
                        }
                    )
                    store["benchmark_results"].append(combo_results[0])
                summary = combo_results[0]
                summary[info["outcome"]] += 1
                summary["cumulative_reward"] += reward
                summary["total_steps"] += info["step"]
                summary["episodes"] += 1
                if summary["episodes"] >= store["num_episodes"]:
                    summary.update(
                        {
                            "escape_rate": round(100.0 * summary["escape"] / summary["episodes"], 1),
                            "caught_rate": round(100.0 * summary["caught"] / summary["episodes"], 1),
                            "timeout_rate": round(100.0 * summary["timeout"] / summary["episodes"], 1),
                            "avg_reward": round(summary["cumulative_reward"] / summary["episodes"], 3),
                            "avg_steps": round(summary["total_steps"] / summary["episodes"], 1),
                        }
                    )
                    store["combo_index"] += 1
                    if store["combo_index"] >= len(store["combos"]):
                        store["status"] = "finished"
                        store["running"] = False
                        summary_metrics = summarize_results(store["benchmark_results"])
                        store["best_combo"] = summary_metrics["best_combo"]
                        store["worst_combo"] = summary_metrics["worst_combo"]
                    else:
                        build_next_episode(store)
                else:
                    build_next_episode(store)
            break

    interval_disabled = not store.get("running")
    if store["status"] == "finished":
        interval_disabled = True
    hint = "Running simulation. Pause to stop updates." if store.get("running") else store.get("status", "")
    return store, interval_disabled, interval_delay, hint


@app.callback(
    Output("mario-strategy", "value"),
    Output("monster-strategy", "value"),
    Output("mario-strategy", "disabled"),
    Output("monster-strategy", "disabled"),
    Input("store-sim-data", "data"),
)
def update_strategy_dropdowns(store: Dict[str, Any]):
    if store is None:
        store = default_state()

    if store["mode"] == "benchmark":
        if not store.get("initialized"):
            return (
                store.get("mario_strategy", "random"),
                store.get("monster_strategy", "aggressive"),
                False,
                False,
            )
        if store.get("combos") and store["combo_index"] < len(store["combos"]):
            combo = store["combos"][store["combo_index"]]
            return combo["mario"], combo["monster"], True, True
        return "random", "aggressive", True, True

    return (
        store.get("mario_strategy", "random"),
        store.get("monster_strategy", "aggressive"),
        False,
        False,
    )


@app.callback(
    Output("grid-figure", "figure"),
    Output("status-text", "children"),
    Output("episode-text", "children"),
    Output("step-text", "children"),
    Output("combo-text", "children"),
    Output("metrics-cards", "children"),
    Output("escape-rate-chart", "figure"),
    Output("reward-chart", "figure"),
    Output("outcome-chart", "figure"),
    Output("steps-chart", "figure"),
    Output("benchmark-table", "data"),
    Output("benchmark-summary", "children"),
    Output("benchmark-bar", "figure"),
    Input("store-sim-data", "data"),
)
def render_dashboard(store: Dict[str, Any]):
    if store is None:
        store = default_state()

    metrics = compute_aggregate_metrics(store)
    cards = build_summary_cards(metrics)
    grid_fig = build_grid_figure(store["grid_data"])

    episode_numbers = [item["episode"] for item in store["episode_history"]]
    escape_series = [100.0 * sum(1 for item in store["episode_history"][: i + 1] if item["outcome"] == "escape") / max(i + 1, 1) for i in range(len(store["episode_history"]))]
    reward_series = [item["reward"] for item in store["episode_history"]]
    steps_series = [item["steps"] for item in store["episode_history"]]

    escape_chart = build_line_chart(episode_numbers, escape_series, "Escape rate over time", "Escape rate (%)")
    reward_chart = build_line_chart(episode_numbers, reward_series, "Reward per episode", "Reward")
    steps_chart = build_line_chart(episode_numbers, steps_series, "Steps per episode", "Steps")
    outcome_chart = build_outcome_chart(store)

    benchmark_data = []
    benchmark_summary = ""
    benchmark_bar = go.Figure()
    if store["mode"] == "benchmark":
        benchmark_data = [
            {
                "label": item["label"],
                "escape_rate": item.get("escape_rate", 0.0),
                "caught_rate": item.get("caught_rate", 0.0),
                "timeout_rate": item.get("timeout_rate", 0.0),
                "avg_reward": item.get("avg_reward", 0.0),
                "avg_steps": item.get("avg_steps", 0.0),
            }
            for item in store["benchmark_results"]
        ]
        if benchmark_data:
            benchmark_bar = build_benchmark_bar(benchmark_data)
            benchmark_summary = html.Div(
                [
                    html.P(f"Best combo: {store['best_combo']}", className="mb-1"),
                    html.P(f"Worst combo: {store['worst_combo']}", className="mb-1"),
                ]
            )
        else:
            benchmark_bar = go.Figure()
            benchmark_bar.update_layout(
                title="Benchmark live update",
                plot_bgcolor="#111111",
                paper_bgcolor="#111111",
                font=dict(color="#ffffff"),
            )

    return (
        grid_fig,
        store.get("status", "ready"),
        str(store.get("current_episode", 0)),
        str(store.get("current_step", 0)),
        store.get("current_combo", ""),
        cards,
        escape_chart,
        reward_chart,
        outcome_chart,
        steps_chart,
        benchmark_data,
        benchmark_summary,
        benchmark_bar,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8080)
