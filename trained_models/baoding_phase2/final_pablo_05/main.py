import copy
import json
import os
import shutil
from calendar import c
from datetime import datetime

import numpy as np
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from src.envs.environment_factory import EnvironmentFactory
from src.metrics.custom_callbacks import EvaluateLSTM
from src.metrics.sb_callbacks import EnvDumpCallback

env_name = "CustomMyoBaodingBallsP2"

# saving criteria
saving_criteria = "score" #score

# whether this is the first task of the curriculum (True) or it is loading a previous task (False)
FIRST_TASK = False

# Path to normalized Vectorized environment (if not first task)
PATH_TO_NORMALIZED_ENV = "output/training/2022-10-31/10-15-45/training_env.pkl"  # "trained_models/normalized_env_original"

# Path to pretrained network (if not first task)
PATH_TO_PRETRAINED_NET = "output/training/2022-10-31/10-15-45/best_model.zip"  # "trained_models/best_model.zip"

# Tensorboard log (will save best model during evaluation)
now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join("output", "training", now)


# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 0,
        "pos_dist_2": 0,
        "act_reg": 0,
        "alive": 0,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    "enable_rsi": False,
    "rsi_probability": 0,
    'balls_overlap': False,
    "overlap_probability": 0,
    "limit_init_angle": np.pi,
    "goal_time_period": [4, 6],   # phase 2: (4, 6)
    "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
    # Randomization in physical properties of the baoding balls
    'obj_size_range': (0.018, 0.024),    #(0.018, 0.024)   # Object size range. Nominal 0.022
    'obj_mass_range': (0.030, 0.300),    #(0.030, 0.300)   # Object weight range. Nominal 43 gms
    'obj_friction_change': (0.2, 0.001, 0.00002), # (0.2, 0.001, 0.00002) nominal: 1.0, 0.005, 0.0001
    'task_choice': 'random'
}

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, env_config, num_env, start_index=0):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.register(env_name, **env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    with open(os.path.join(TENSORBOARD_LOG, "config.json"), "w") as file:
        json.dump(config, file)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    # Create vectorized environments:
    envs = make_parallel_envs(env_name, config, num_env=16)

    # Normalize environment:
    if FIRST_TASK:
        envs = VecNormalize(envs)
    else:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)


    # Callbacks for score and for effort


    config_score = {
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "enable_rsi": False,
        "rsi_probability": 0,
        'balls_overlap': False,
        "overlap_probability": 0,
        "limit_init_angle": np.pi,
        "goal_time_period": [4, 6],   # phase 2: (4, 6)
        "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
        "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
        # Randomization in physical properties of the baoding balls
        'obj_size_range': (0.018, 0.024),    #(0.018, 0.024)   # Object size range. Nominal 0.022
        'obj_mass_range': (0.030, 0.300),    #(0.030, 0.300)   # Object weight range. Nominal 43 gms
        'obj_friction_change': (0.2, 0.001, 0.00002), # (0.2, 0.001, 0.00002) nominal: 1.0, 0.005, 0.0001
        'task_choice': 'random'
    }

    config_effort = {
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 1,
            "alive": 0,
            "solved": 0,
            "done": 0,
            "sparse": 0,
        },
        "enable_rsi": False,
        "rsi_probability": 0,
        'balls_overlap': False,
        "overlap_probability": 0,
        "limit_init_angle": np.pi,
        "goal_time_period": [4, 6],   # phase 2: (4, 6)
        "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
        "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
        # Randomization in physical properties of the baoding balls
        'obj_size_range': (0.018, 0.024),    #(0.018, 0.024)   # Object size range. Nominal 0.022
        'obj_mass_range': (0.030, 0.300),    #(0.030, 0.300)   # Object weight range. Nominal 43 gms
        'obj_friction_change': (0.2, 0.001, 0.00002), # (0.2, 0.001, 0.00002) nominal: 1.0, 0.005, 0.0001
        'task_choice': 'random'
    }

    env_score = EnvironmentFactory.register(env_name, **config_score)
    env_effort = EnvironmentFactory.register(env_name, **config_effort)

    score_callback = EvaluateLSTM(eval_freq = 200000, eval_env = env_score, name = 'eval/score', num_episodes=10)
    effort_callback = EvaluateLSTM(eval_freq = 200000, eval_env = env_effort, name = 'eval/effort', num_episodes=10)

    # Evaluation Callback

    # Create vectorized environments:
    if saving_criteria=="score":
        eval_envs = make_parallel_envs(env_name, config_score, num_env=16)
    elif saving_criteria=="dense_rewards":
        eval_envs = make_parallel_envs(env_name, config, num_env=16)
    else:
        raise ValueError('Unrecognized saving criteria')

    if FIRST_TASK:
        eval_envs = VecNormalize(eval_envs)
    else:
        eval_envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, eval_envs) 

    env_dump_callback = EnvDumpCallback(TENSORBOARD_LOG, verbose=0)

    eval_callback = EvalCallback(
        eval_envs,
        callback_on_new_best=env_dump_callback,
        best_model_save_path=TENSORBOARD_LOG,
        log_path=TENSORBOARD_LOG,
        eval_freq=7500,
        deterministic=True,
        render=False,
        n_eval_episodes=100,
    )

    # Create model (hyperparameters from RL Zoo HalfCheetak)
    if FIRST_TASK:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            envs,
            verbose=2,
            tensorboard_log=TENSORBOARD_LOG,
            batch_size=32,
            n_steps=512,
            gamma=0.99,
            gae_lambda=0.9,
            n_epochs=10,
            ent_coef= 3e-6,
            learning_rate=2e-5,
            clip_range=0.25,
            use_sde=True,
            max_grad_norm=0.8,
            vf_coef=0.5,
            policy_kwargs=dict(
                log_std_init=-2,
                ortho_init=False,
                activation_fn=nn.ReLU,
                net_arch=[dict(pi=[], vf=[])],
                enable_critic_lstm=True,
                lstm_hidden_size=128,
            ),
        )
    else:
        custom_objects = {      # need to define this since my python version is newer
        "lr_schedule": lambda _: 1.5e-04,
        "learning_rate": lambda _: 1.5e-04,
        "clip_range": 0.2,
        "n_steps": 4096,
        "batch_size": 4096,
        "ent_coef": 0.00025,
        }
        model = RecurrentPPO.load(
            PATH_TO_PRETRAINED_NET,
            env=envs,
            tensorboard_log=TENSORBOARD_LOG,
            device='cuda',
            custom_objects=custom_objects
        )

    # Train and save model
    model.learn(
        total_timesteps=30_000_000, callback=[eval_callback,score_callback,effort_callback], reset_num_timesteps=True
    )

    model.save(os.path.join(TENSORBOARD_LOG, "final_model.pkl"))
    envs.save(os.path.join(TENSORBOARD_LOG, "final_env.pkl"))
