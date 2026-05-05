# 05. Training Pipeline

## Training Loop Structure

The training pipeline follows a standard RL episode-based approach:

```python
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        episode_reward += reward
    
    # Logging and checkpointing
    logger.log_episode(episode_reward, info)
    if episode % checkpoint_freq == 0:
        agent.save_checkpoint()
```

## Key Hyperparameters

### Network Parameters
- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 128
- **Hidden Layers**: [128, 64] units
- **Activation**: ReLU
- **Weight Initialization**: Orthogonal

### Exploration Parameters
- **Initial Epsilon**: 1.0
- **Epsilon Decay**: 0.999
- **Minimum Epsilon**: 0.1
- **Decay Frequency**: Per episode

### Training Parameters
- **Buffer Size**: 10,000
- **Target Update Frequency**: 10 episodes
- **Soft Update τ**: 0.001
- **Discount Factor γ**: 0.99
- **Maximum Steps**: 200 per episode

## Logging System

### Metrics Tracked
- **Episode Reward**: Total reward per episode
- **Escape Rate**: Percentage of successful escapes
- **Caught Rate**: Percentage caught by monster
- **Timeout Rate**: Percentage of episodes timing out
- **Average Episode Length**: Steps per episode
- **Loss Values**: Q-network training loss

### Logging Frequency
- **Per Episode**: Reward, success/failure counts
- **Per 100 Episodes**: Rolling averages, escape rates
- **Per 1000 Episodes**: Full statistics, model checkpoints

## Checkpoint System

### Automatic Checkpoints
- **Frequency**: Every 1000 episodes
- **Components Saved**:
  - Q-network weights
  - Target network weights
  - Optimizer state
  - Replay buffer (optional)
  - Training statistics
- **Naming Convention**: `dqn_checkpoint_ep{episode}.pt`

### Best Model Tracking
- **Criteria**: Highest escape rate over evaluation window
- **File**: `dqn_best.pt`
- **Update Condition**: New best performance detected

## Early Stopping

### Plateau Detection
- **Window Size**: 500 episodes
- **Improvement Threshold**: 0.5% escape rate improvement
- **Patience**: 2000 episodes without improvement
- **Action**: Increase exploration (epsilon boost) or terminate training

### Convergence Criteria
- **Target Performance**: 75% escape rate
- **Stability Window**: 1000 episodes of consistent performance
- **Termination**: Training stops when criteria met or max episodes reached

## Training Phases

1. **Warmup Phase** (Episodes 1-1000)
   - High exploration (epsilon decay active)
   - Buffer filling and initial learning

2. **Learning Phase** (Episodes 1000-5000)
   - Balanced exploration/exploitation
   - Network convergence and policy refinement

3. **Fine-tuning Phase** (Episodes 5000+)
   - Low exploration (epsilon near minimum)
   - Performance optimization and stabilization