import gym
import os
import myosuite
import numpy as np
from definitions import ROOT_DIR


myosuite_path = os.path.join(ROOT_DIR, "data", "myosuite")

# MyoChallenge Baoding: Phase1 env
gym.envs.registration.register(
    id="CustomMyoChallengeBaodingP1-v1",
    entry_point="src.envs.baoding:CustomBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_baoding.mjb",
        "normalize_act": True,
        # 'goal_time_period': (5, 5),
        # "goal_xrange": (0.025, 0.025),
        # "goal_yrange": (0.028, 0.028),
    },
)

# MyoChallenge Baoding: env with history
gym.envs.registration.register(
    id="HistoryMyoChallengeBaodingP1-v1",
    entry_point="src.envs.baoding:HistoryBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_baoding.mjb",
        "normalize_act": True,
        # 'goal_time_period': (5, 5),
        # "goal_xrange": (0.025, 0.025),
        # "goal_yrange": (0.028, 0.028),
    },
)

# MyoChallenge Baoding: Phase2 env
gym.envs.registration.register(
    id="CustomMyoChallengeBaodingP2-v1",
    entry_point="src.envs.baoding:CustomBaodingP2Env",
    max_episode_steps=200,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_baoding.mjb",
        "normalize_act": True,
        "goal_time_period": (4, 6),
        "goal_xrange": (0.020, 0.030),
        "goal_yrange": (0.022, 0.032),
        # Randomization in physical properties of the baoding balls
        "obj_size_range": (0.018, 0.024),  # Object size range. Nominal 0.022
        "obj_mass_range": (0.030, 0.300),  # Object weight range. Nominal 43 gms
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
        "task_choice": "random",
    },
)

# MyoChallenge Die: Phase1 env
gym.envs.registration.register(
    id="CustomMyoChallengeDieReorientP1-v0",
    entry_point="src.envs.reorient:CustomReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_die.mjb",
        "normalize_act": True,
        "frame_skip": 5,
        "goal_pos": (-0.010, 0.010),  # +- 1 cm
        "goal_rot": (-1.57, 1.57),  # +-90 degrees
    },
)

# MyoChallenge Die: Goal env
gym.envs.registration.register(
    id="GoalHistoryMyoChallengeDieReorientP2-v0",
    entry_point="src.envs.reorient:GoalHistoryReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": myosuite_path + "/assets/hand/myo_hand_die.mjb",
        "normalize_act": True,
        "frame_skip": 5,
        # Randomization in goals
        "goal_pos": (-0.020, 0.020),  # +- 2 cm
        "goal_rot": (-3.14, 3.14),  # +-180 degrees
        # Randomization in physical properties of the die
        "obj_size_change": 0.007,  # +-7mm delta change in object size
        "obj_friction_change": (0.2, 0.001, 0.00002),  # nominal: 1.0, 0.005, 0.0001
    },
)
