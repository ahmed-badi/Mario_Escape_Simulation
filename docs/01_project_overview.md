# 01. Project Overview

## Project Description

Mario Escape Simulation is a reinforcement learning (RL) research project implementing a grid-based escape scenario where an agent (Mario) must navigate to an exit while avoiding a pursuing monster. The project combines a custom Gym-like environment with a Deep Q-Network (DQN) agent, providing a complete RL pipeline for training, evaluation, and analysis.

## Objectives

- Develop a modular RL environment for grid-based pursuit-evasion scenarios
- Implement and optimize a DQN agent for escape behavior learning
- Compare RL performance against classical pathfinding algorithms (Greedy, A*)
- Generate trajectory datasets for supervised learning approaches
- Provide a research-grade codebase for RL experimentation

## Problem Definition

The environment consists of a 2D grid where:
- **Mario** (agent) starts at a random position and must reach any exit
- **Monster** (adversary) pursues Mario using configurable strategies
- **Grid** contains walls, open spaces, and exit points
- **Episode** ends when Mario escapes (+10 reward), gets caught (-10 reward), or times out (0 reward)

This represents a classic pursuit-evasion problem with sparse rewards, making it challenging for RL algorithms to learn optimal escape policies.

## Why Reinforcement Learning?

RL is employed because:
- The optimal escape strategy depends on dynamic monster behavior
- Classical algorithms (A*, Greedy) assume static environments
- RL can learn adaptive policies that account for monster movement patterns
- The problem involves sequential decision-making under uncertainty

## Key Features

- **Modular Environment**: Configurable grid sizes, monster strategies, reward structures
- **Multiple Agent Types**: DQN, Greedy, A*, Random baselines
- **Comprehensive Training Pipeline**: Experience replay, target networks, epsilon-greedy exploration
- **Evaluation Framework**: Multi-strategy comparison with statistical analysis
- **Dataset Generation**: Trajectory collection for ML training
- **Visualization Tools**: Grid rendering, training curves, strategy comparisons