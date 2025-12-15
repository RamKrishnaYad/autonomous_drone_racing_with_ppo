# autonomous_drone_racing_with_ppo

Overview

This repository contains a complete End-to-End Deep Reinforcement Learning (DRL) framework for autonomous drone racing. The goal of this project is to train a quadrotor agent to fly through a complex 8-gate circuit at high speeds, using only onboard state observations.

Unlike classical control methods (PID/MPC) that rely on simplified physics models, this project uses Proximal Policy Optimization (PPO) to learn a robust control policy directly from interaction with a high-fidelity MuJoCo physics simulation. To overcome the exploration challenges of agile flight, we implement a three-stage Curriculum Learning strategy.

# Key Features

Physics Engine: High-fidelity rigid body dynamics using MuJoCo.

Algorithm: Custom implementation of PPO (Proximal Policy Optimization).

Curriculum Learning: 3-Phase training pipeline (Hover to Waypoint to Racing).

Reward Shaping: Novel differential progress reward to solve the "gate loitering" local optimum.

Dynamics Interface: "LQR-style" mixer layer bridging abstract neural net actions to physical motor thrusts.

# Installation

This project requires Python 3.8+ and MuJoCo.

1. Clone the Repository

git clone [https://github.com/RamKrishnaYad/autonomous_drone_racing_with_ppo.git]


2. Create Environment (macOS)

python -m venv drone_rl
Source drone_rl/bin/activate 

3. Install Dependencies

pip install gymnasium[mujoco]
pip install torch torchvision numpy pandas matplotlib

# Usage

The agent is trained using a Curriculum Learning strategy. You must train the phases in order, as each phase saves a model checkpoint that the next phase loads.

Phase 1: Hover Stabilization

Teaches the drone to fight gravity and maintain a stable orientation.

python train_hover.py
# Output: Saves 'ppo_quad_hover.pt'

Phase 2: Waypoint Navigation

Teaches the drone to tilt and generate velocity to reach moving targets.

python train_waypoint.py
# Loads 'ppo_quad_hover.pt' -> Saves 'ppo_quad_waypoint.pt'

Phase 3: Continuous Racing (The Final Task)

Fine-tunes the agent to navigate the 8-gate race track efficiently.

python train_racing.py
# Loads 'ppo_quad_waypoint.pt' -> Saves 'ppo_quad_racing.pt'

Evaluating the policy

python eval_ppo_waypoint.py
# Evaluates the waypoint policy " ppo_quad_waypoint" in the Mujoco

python eval_ppo_waypoint.py
# Evaluates the racing policy " ppo_quad_wracing" in the Mujoco


# System Architecture

The system is designed as a hierarchical control loop:

Input (Observation): 12D State Vector (Relative Position, Linear Velocity, Rotation Matrix, Angular Velocity).

The Brain (PPO Agent): An Actor-Critic neural network (MLP, 64x64 units) that maps the state to normalized actions.

The Reflexes (Dynamics Interface): Maps normalized actions $[-1, 1]$ to physical motor thrusts $[0, 4N]$ and applies torques to the rigid body.

The World (MuJoCo): Simulates the physics (Gravity, Drag, Collisions) and returns the new state and reward.

# Reward Function

We use a Differential Progress Reward to encourage velocity:
$$ R_t = 5.0 \cdot (d_{t-1} - d_t) + R_{bonus} + R_{penalties} $$

# Results

Racing Performance

The agent successfully completes 3 full laps without crashing, demonstrating emergent behaviors like:

Banking: Rolling into turns before reaching them.

Apexing: Cutting corners to minimize distance.

Continuous Thrust: Maintaining forward momentum through gates.

# File Structure

├── drone.xml                   # MuJoCo Physics Model (Quadrotor & Track)
├── drone_model.py              # Dynamics Interface (Thrust Mixing & Physics Math)
├── drone_env.py                # Gym Environment Wrapper (Reward Logic & State Obs)
├── ppo_agent.py                # PPO Algorithm Implementation (Actor-Critic NN)
├── train_hover.py              # Phase 1 Training Script
├── train_waypoint.py           # Phase 2 Training Script
├── train_racing.py             # Phase 3 Training Script
├── eval_ppo_waypoint.py.py     # Evaluates the waypoint policy in Mujoco
├── eval_ppo_racing.py.py       # Evaluates the racing policy in Mujoco
└── README.md                   # This file


# References

Inspiration: Kaufmann et al., "Champion-level drone racing using deep reinforcement learning", Nature 2023.

Survey: Hanover et al., "Autonomous Drone Racing: A Survey", 2023.

# Author: Ram Krishna Yadav
# Contact: ramkrishyad@gmail.com  / https://www.linkedin.com/in/ramkrishnayadav19/
