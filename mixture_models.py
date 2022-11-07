import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from src.envs.environment_factory import EnvironmentFactory

# evaluation parameters:
render = False
num_episodes = 2_000

env_name = "MyoBaodingBallsP2"


#paths
load_folders = [
    "trained_models/baoding_phase2/final_pablo_511",
    "trained_models/baoding_phase2/alberto_518"]

PATH_TO_NORMALIZED_ENV = [load_folder + "/training_env.pkl" for load_folder in load_folders]
PATH_TO_PRETRAINED_NET = [load_folder + "/best_model.zip" for load_folder in load_folders]



# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 0,
        "pos_dist_2": 0,
        "act_reg": 0,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    "goal_time_period": [4, 6],   # phase 2: (4, 6)
    "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
    "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
    # Randomization in physical properties of the baoding balls
    'obj_size_range': (0.018, 0.024),    #(0.018, 0.024)   # Object size range. Nominal 0.022
    'obj_mass_range': (0.030, 0.300),    #(0.030, 0.300)   # Object weight range. Nominal 43 gms
    'obj_friction_change': (0.2, 0.001, 0.00002), # (0.2, 0.001, 0.00002)
    'task_choice': 'random'
}


# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, env_config, num_env, start_index=0):   # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # Create vectorized environments:
    envs = [make_parallel_envs(env_name, config, num_env=1) for _ in range(len(load_folders))]
    
    envs = [VecNormalize.load(path, envs[i]) for i,path in enumerate(PATH_TO_NORMALIZED_ENV)]
    for env in envs:
        env.training = False
        env.norm_reward = False

    # Create model
    custom_objects = {
        "lr_schedule": 0,
        "clip_range": 0,
    }

    eval_models = []
    for i in range(len(envs)):
        eval_models.append(
            RecurrentPPO.load(
                PATH_TO_PRETRAINED_NET[i],
                env=envs[i])
        )

    # EVALUATE
    eval_env = EnvironmentFactory.create(env_name, **config)

    # Enjoy trained agent
    perfs = []
    lens = []
    for i in range(num_episodes):
        all_lstm_states = [None]*len(eval_models)
        cum_rew = 0
        step = 0
        # eval_env.reset()
        # eval_env.step(np.zeros(39))
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            if render:
                eval_env.sim.render(mode="window")
            
            all_actions = []
            for count,eval_model in enumerate(eval_models):
                action, all_lstm_states[count] = eval_model.predict(
                    envs[count].normalize_obs(obs),
                    state= all_lstm_states[count],
                    episode_start=episode_starts,
                    deterministic=True,
                )
                all_actions.append(action)

            final_action = np.mean(all_actions,axis=0)

            obs, rewards, done, info = eval_env.step(final_action)
            episode_starts = done
            cum_rew += rewards
            step += 1
        lens.append(step)
        perfs.append(cum_rew)
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)

        if (i + 1) % 10 == 0:
            print(f"\nEpisode {i+1}/{num_episodes}")
            print(f"Average len: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {np.std(perfs):.2f}\n")