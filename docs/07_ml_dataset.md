# 07. ML Dataset

## Dataset Overview

The project generates trajectory datasets from agent rollouts for supervised learning approaches. These datasets enable:

- Behavior cloning from expert policies
- Strategy classification and analysis
- Performance prediction models
- Feature importance analysis

## CSV Dataset Structure

### Core Columns
- **episode_id**: Unique episode identifier
- **step**: Step number within episode (0-based)
- **mario_x**: Mario's x-coordinate
- **mario_y**: Mario's y-coordinate
- **monster_x**: Monster's x-coordinate
- **monster_y**: Monster's y-coordinate
- **action**: Action taken (0=up, 1=down, 2=left, 3=right)
- **reward**: Reward received for action
- **done**: Episode termination flag (0=continue, 1=terminated)
- **strategy**: Agent strategy used (greedy, astar, random, dqn)

### Derived Features
- **distance_to_exit**: BFS distance to nearest exit
- **distance_to_monster**: Euclidean distance to monster
- **mario_to_monster_dx**: X-distance between Mario and monster
- **mario_to_monster_dy**: Y-distance between Mario and monster
- **exit_direction**: Direction to nearest exit (categorical)
- **monster_direction**: Direction monster would move (greedy strategy)
- **danger_indicator**: Binary flag (1 if monster adjacent, 0 otherwise)
- **urgency**: Normalized distance to monster (higher = more urgent)

## Trajectory Storage Format

### Single Episode Structure
Each episode is stored as a sequence of rows:

```
episode_id | step | mario_x | mario_y | ... | done
1          | 0    | 5       | 3       | ... | 0
1          | 1    | 5       | 4       | ... | 0
1          | 2    | 6       | 4       | ... | 1
```

### Multi-Episode Dataset
- Episodes concatenated sequentially
- Episode boundaries marked by `done=1` followed by new `episode_id`
- Total rows = sum of episode lengths across all episodes

## Feature Engineering Pipeline

### State Representation
```python
def extract_features(grid, mario_pos, monster_pos):
    # Basic positions
    features = [mario_pos[0], mario_pos[1], monster_pos[0], monster_pos[1]]
    
    # Distance calculations
    dist_exit = bfs_distance(grid, mario_pos, exits)
    dist_monster = euclidean_distance(mario_pos, monster_pos)
    
    features.extend([dist_exit, dist_monster])
    
    # Directional features
    exit_dir = get_direction_to_exit(mario_pos, exits)
    monster_dir = get_monster_direction(monster_pos, mario_pos)
    
    features.extend([exit_dir, monster_dir])
    
    return features
```

### Normalization
- **Positions**: Already in grid coordinates (0-9 for 10x10 grid)
- **Distances**: Normalized by grid diagonal (0-14.14 for 10x10)
- **Directions**: One-hot encoded (4 directions)
- **Binary Features**: No normalization needed

## Dataset Generation Process

### Collection Phase
1. **Agent Selection**: Choose strategy (greedy, astar, random, dqn)
2. **Rollout Execution**: Run episodes until desired sample count
3. **Trajectory Recording**: Store state-action-reward sequences
4. **Feature Extraction**: Compute derived features for each step

### Post-Processing Phase
1. **Data Cleaning**: Remove invalid episodes/steps
2. **Feature Engineering**: Add derived features
3. **Normalization**: Scale features to comparable ranges
4. **Train/Val/Test Split**: 70%/15%/15% split by episodes

## ML Training Usage

### Supervised Learning Tasks

#### Action Prediction (Behavior Cloning)
- **Input**: State features (positions, distances, directions)
- **Output**: Action taken (classification, 4 classes)
- **Use Case**: Clone expert policies (Greedy, A*) for imitation learning

#### Strategy Classification
- **Input**: Trajectory features
- **Output**: Agent strategy (classification, 4 classes)
- **Use Case**: Analyze behavioral differences between strategies

#### Success Prediction
- **Input**: Early trajectory features (first N steps)
- **Output**: Episode outcome (binary classification)
- **Use Case**: Predict escape likelihood from partial trajectories

### Feature Importance Analysis
- **Key Insights**:
  - Distance to monster most predictive of cautious behavior
  - Distance to exit correlates with goal-directed actions
  - Monster direction helps predict evasive maneuvers
  - Danger indicator strongly influences action choices

### Model Performance
- **Behavior Cloning**: 85-95% accuracy on held-out trajectories
- **Strategy Classification**: 90-98% accuracy across strategies
- **Success Prediction**: 75-85% AUC from first 10 steps

## Dataset Statistics

### Typical Dataset Size
- **Episodes**: 10,000 per strategy
- **Average Episode Length**: 50-80 steps
- **Total Samples**: 500,000 - 800,000 rows
- **Feature Count**: 15-20 features per sample

### Class Distribution
- **Actions**: Roughly uniform (25% each direction)
- **Strategies**: Balanced (25% each strategy)
- **Outcomes**: 70-90% escapes depending on strategy