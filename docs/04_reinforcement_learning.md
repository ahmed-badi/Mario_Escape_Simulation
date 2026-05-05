# 04. Reinforcement Learning

## DQN Algorithm Overview

The project implements Deep Q-Network (DQN) with several enhancements:

- **Q-Network**: Neural network approximating Q-values for state-action pairs
- **Experience Replay**: Buffer storing past experiences for stable learning
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy Exploration**: Balance between exploitation and exploration

## Core Components

### Q-Network Architecture
- **Input**: State representation (grid + positions)
- **Hidden Layers**: 2 fully-connected layers (128, 64 units)
- **Output**: Q-values for 4 actions (up, down, left, right)
- **Activation**: ReLU for hidden layers, linear for output

### Experience Replay
- **Buffer Size**: 10,000 experiences
- **Batch Size**: 128 samples per update
- **Sampling**: Uniform random from buffer
- **Stabilization**: Breaks correlation between consecutive experiences

### Epsilon-Greedy Strategy
- **Initial Epsilon**: 1.0 (full exploration)
- **Decay Rate**: 0.999 per episode
- **Minimum Epsilon**: 0.1 (maintains some exploration)
- **Decay Schedule**: Exponential decay over training

### Target Network Updates
- **Update Frequency**: Every 10 episodes
- **Soft Updates**: τ = 0.001 for gradual synchronization
- **Purpose**: Prevents oscillating Q-value targets

## Algorithm Limitations

DQN faces several challenges in this environment:

- **Sparse Rewards**: Only +10/-10 at episode end, difficult credit assignment
- **High Dimensionality**: Grid state space grows exponentially with size
- **Temporal Credit Assignment**: Long sequences between actions and rewards
- **Overestimation Bias**: Q-values tend to be optimistic
- **Sample Inefficiency**: Requires many episodes for convergence

## Comparison with Baseline Algorithms

| Algorithm | Escape Rate | Path Optimality | Adaptability | Training Required |
|-----------|-------------|-----------------|--------------|-------------------|
| **Greedy** | ~85% | High (BFS optimal) | Low (static) | None |
| **A*** | ~90% | High (heuristic optimal) | Low (static) | None |
| **Random** | ~10% | Low | High (random) | None |
| **DQN** | ~60-75% | Medium | High (learned) | Extensive |

### Why DQN Underperforms Greedy/A*

1. **Approximation Error**: Neural network cannot perfectly represent optimal Q-function
2. **Exploration Challenges**: Epsilon-greedy may not discover optimal paths consistently
3. **Reward Sparsity**: Limited feedback makes learning inefficient
4. **Monster Adaptation**: Greedy algorithms exploit monster predictability perfectly
5. **Convergence Issues**: DQN may converge to suboptimal policies

### DQN Advantages

- **Generalization**: Learns patterns applicable to different grid layouts
- **Adaptability**: Can adjust to changing monster strategies
- **Scalability**: Neural networks handle larger state spaces than explicit search
- **Transfer Learning**: Trained policies can adapt to similar environments