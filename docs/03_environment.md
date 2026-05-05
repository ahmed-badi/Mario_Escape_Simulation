# 03. Environment

## Grid System

The environment operates on a 2D grid with the following components:

- **Dimensions**: Configurable size (default 10x10)
- **Cell Types**:
  - `0`: Empty space (navigable)
  - `1`: Wall (impassable)
  - `2`: Exit (goal state)
- **Boundaries**: Grid edges are walls
- **Connectivity**: 4-directional movement (up, down, left, right)

## Mario Mechanics

Mario represents the learning agent with the following properties:

- **Starting Position**: Random empty cell (not adjacent to monster)
- **Movement**: Discrete actions (4 directions)
- **Constraints**:
  - Cannot move through walls
  - Invalid moves result in position penalty
  - Movement is deterministic (no stochasticity)

## Monster Behavior

The monster pursues Mario using configurable strategies:

- **Random**: Uniform random movement
- **Greedy**: Moves toward Mario using BFS (shortest path)
- **Semi-Aggressive**: Greedy with 30% random exploration
- **Movement**: Same 4-directional constraints as Mario

## Reward System

The reward structure is designed to guide escape behavior:

- **Escape**: +10.0 (reaching any exit)
- **Caught**: -10.0 (monster reaches Mario)
- **Timeout**: 0.0 (maximum steps exceeded)
- **Invalid Move**: -0.1 (attempting to move into wall)
- **Step Penalty**: -0.02 (encourages efficient paths)
- **Distance Shaping**: +0.3 × (previous_distance - current_distance) (potential-based reward)

## Episode Lifecycle

Each episode follows this sequence:

1. **Initialization**
   - Generate random grid layout
   - Spawn Mario at random position
   - Spawn monster at random position (minimum distance constraint)
   - Reset step counter and reward accumulator

2. **Interaction Loop**
   - Agent observes current state
   - Agent selects action
   - Environment executes action
   - Calculate reward and check termination conditions
   - Update positions and step counter

3. **Termination Conditions**
   - Mario reaches exit (success)
   - Monster catches Mario (failure)
   - Maximum steps reached (timeout, default 200)
   - Invalid state (should not occur in normal operation)

4. **Reset**
   - Clear grid state
   - Generate new random layout
   - Return final reward and episode statistics