# 02. Architecture

## System Architecture

The project follows a modular architecture separating concerns into distinct layers:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Environment   │◄──►│     Agents      │◄──►│    Training     │
│   (Grid, RL)    │    │   (DQN, etc.)   │    │   (Pipeline)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Evaluation    │    │       ML        │    │     Data        │
│   (Comparison)  │    │   (Supervised)  │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### Environment Layer
- **Grid System**: 2D grid representation with walls, exits, and dynamic entities
- **RL Interface**: Gym-compatible environment with step/reset methods
- **State Management**: Position tracking, collision detection, episode lifecycle

### Agent Layer
- **DQN Agent**: PyTorch-based neural network with experience replay
- **Baseline Agents**: Greedy (BFS), A* (heuristic search), Random (uniform actions)
- **Strategy Interface**: Unified API for different agent implementations

### Training Layer
- **Training Loop**: Episode-based training with logging and checkpointing
- **Experience Replay**: Buffer for stabilizing Q-learning updates
- **Hyperparameter Management**: Configurable learning rates, exploration schedules

### Evaluation Layer
- **Multi-Agent Comparison**: Statistical analysis across strategies
- **Metrics Collection**: Escape rates, episode lengths, reward distributions
- **Visualization**: Training curves, grid animations, performance heatmaps

### ML Pipeline Layer
- **Dataset Generation**: Trajectory collection from agent rollouts
- **Feature Engineering**: State-action-reward sequences for supervised learning
- **Model Training**: Scikit-learn classifiers for behavior prediction

## Data Flow

1. **Environment Initialization**
   - Grid generation → Mario/Monster spawning → Initial state

2. **Agent Interaction**
   - State observation → Action selection → Environment step → Reward/Next state

3. **Training Loop**
   - Experience storage → Batch sampling → Network update → Target network sync

4. **Evaluation**
   - Agent rollout → Metrics calculation → Statistical comparison

5. **Dataset Creation**
   - Trajectory collection → Feature extraction → CSV export → ML training

## Key Design Decisions

- **Modular Separation**: Each component can be modified independently
- **Gym Compatibility**: Standard RL interface for easy integration
- **PyTorch Backend**: GPU acceleration and automatic differentiation
- **CPU-First Design**: No external RL libraries, pure PyTorch implementation
- **Research-Friendly**: Extensive logging and visualization for analysis