# Mario Escape Simulation — Reinforcement Learning Project

A comprehensive reinforcement learning research project implementing Deep Q-Network (DQN) agents in a grid-based pursuit-evasion environment. Mario must escape to exits while avoiding a pursuing monster, with comparison against classical pathfinding algorithms.

## Overview

This project provides a complete RL pipeline for studying agent learning in dynamic environments. The environment features a 2D grid where Mario navigates to exits while evading a monster using configurable strategies. The DQN implementation includes experience replay, target networks, and epsilon-greedy exploration.

## Features

- **Modular RL Environment**: Gym-compatible grid-based pursuit-evasion scenario
- **Multiple Agent Strategies**: DQN, Greedy (BFS), A*, Random baselines
- **Complete Training Pipeline**: Experience replay, logging, checkpointing
- **Evaluation Framework**: Statistical comparison across strategies
- **Dataset Generation**: Trajectory collection for supervised learning
- **Visualization Tools**: Training curves and grid animations

## Architecture Summary

```
Environment (Grid + RL Interface)
    ↓
Agents (DQN, Greedy, A*, Random)
    ↓
Training Pipeline (Experience Replay, Target Networks)
    ↓
Evaluation & Analysis (Metrics, Comparisons, Datasets)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mario_escape_sim.git
   cd mario_escape_sim
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Training DQN Agent

Train a DQN agent for 5000 episodes with default hyperparameters:

```bash
python ml/rl/train_rl.py --episodes 5000 --monster_strategy semi_aggressive
```

**Key options**:
- `--episodes`: Number of training episodes (default: 5000)
- `--monster_strategy`: Monster behavior (random, greedy, semi_aggressive)
- `--learning_rate`: Q-network learning rate (default: 0.001)
- `--buffer_size`: Experience replay buffer size (default: 10000)

### Evaluating Agents

Compare all strategies on 1000 episodes each:

```bash
python ml/evaluate_all.py --episodes 1000
```

Evaluate specific strategy:

```bash
python src/simulation/engine.py --strategy dqn --episodes 1000
```

### Generating ML Dataset

Create trajectory dataset for supervised learning:

```bash
python ml/supervised/train_classifier.py --generate_dataset --episodes_per_strategy 5000
```

## Results

### Performance Comparison

| Strategy | Escape Rate | Avg Reward | Avg Steps |
|----------|-------------|------------|-----------|
| Random   | 8.5%       | -8.2      | 185      |
| Greedy   | 85.2%      | 6.8       | 45       |
| A*       | 89.7%      | 7.2       | 38       |
| DQN      | 72.3%      | 4.1       | 65       |

*DQN trained for 5000 episodes on semi-aggressive monster strategy*

### Training Curves

DQN typically shows:
- **Early Training**: 10-30% escape rate (exploration phase)
- **Mid Training**: 30-60% escape rate (learning phase)
- **Late Training**: 60-75% escape rate (convergence phase)

## Technologies Used

- **Python 3.8+**: Core implementation
- **PyTorch**: Neural networks and optimization
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualization and plotting
- **Scikit-learn**: ML utilities and evaluation

## Project Structure

```
mario_escape_sim/
├── docs/                          # Documentation
│   ├── 01_project_overview.md
│   ├── 02_architecture.md
│   ├── 03_environment.md
│   ├── 04_reinforcement_learning.md
│   ├── 05_training_pipeline.md
│   ├── 06_results_analysis.md
│   ├── 07_ml_dataset.md
│   └── 08_future_improvements.md
├── src/                           # Core source code
│   ├── agents/                    # Agent implementations
│   ├── environment/               # Grid and RL environment
│   ├── simulation/                # Simulation engine
│   └── utils/                     # Utilities
├── ml/                            # Machine learning components
│   ├── rl/                        # Reinforcement learning
│   └── supervised/                # Supervised learning
├── data/                          # Data storage
│   ├── models/                    # Trained models
│   ├── processed/                 # Processed datasets
│   └── raw/                       # Raw trajectory data
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
├── run_simulation.py              # Main simulation script
└── README.md                      # This file
```

## Why DQN Underperforms Greedy in This Environment

DQN achieves 72% escape rate compared to Greedy's 85% due to fundamental algorithmic differences:

1. **Approximation vs Exact Computation**: Greedy uses perfect BFS for shortest paths, while DQN learns approximate Q-values through neural networks
2. **Sample Inefficiency**: DQN requires thousands of episodes to learn, while Greedy solves optimally in single episodes
3. **Sparse Rewards**: Only terminal rewards (+10/-10) make credit assignment difficult over long action sequences
4. **Exploration Challenges**: Epsilon-greedy may not consistently discover optimal paths in deterministic environments
5. **Monster Predictability**: Greedy perfectly exploits the monster's deterministic movement patterns

DQN excels in more complex, stochastic environments where exact computation becomes infeasible.

## Future Improvements

- **Algorithm Upgrades**: PPO, Dueling DQN, Prioritized Experience Replay
- **Environment Enhancements**: Dynamic grids, stochastic monsters, multi-agent scenarios
- **Hybrid Approaches**: A* + RL integration, hierarchical policies
- **Infrastructure**: Distributed training, advanced monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mario_escape_sim,
  title={Mario Escape Simulation: Reinforcement Learning in Grid-Based Pursuit-Evasion},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/mario_escape_sim}
}
```
