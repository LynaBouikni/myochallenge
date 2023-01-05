import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.pen_v0 import PenTwirlRandomEnvV0
from myosuite.utils.quat_math import euler2quat


class CustomPenEnv(PenTwirlRandomEnvV0):
    def get_reward_dict(self, obs_dict):
        rwd_dict = super().get_reward_dict(obs_dict)
        pos_align_diff = (
            self.pos_align + rwd_dict["pos_align"]
        )  # the one in the reward has opposite sign
        rot_align_diff = (
            rwd_dict["rot_align"] - self.rot_align
        )  # cosine similarity has opposite sign
        rwd_dict.update(
            {"pos_align_diff": pos_align_diff, "rot_align_diff": rot_align_diff}
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def _setup(
        self,
        obs_keys: list = PenTwirlRandomEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = PenTwirlRandomEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        goal_orient_range=(-1, 1),
        enable_rsi=False,
        rsi_distance=0,
        **kwargs,
    ):
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id("S_grasp")
        self.obj_bid = self.sim.model.body_name2id("Object")
        self.eps_ball_sid = self.sim.model.site_name2id("eps_ball")
        self.obj_t_sid = self.sim.model.site_name2id("object_top")
        self.obj_b_sid = self.sim.model.site_name2id("object_bottom")
        self.tar_t_sid = self.sim.model.site_name2id("target_top")
        self.tar_b_sid = self.sim.model.site_name2id("target_bottom")
        self.pen_length = np.linalg.norm(
            self.sim.model.site_pos[self.obj_t_sid]
            - self.sim.model.site_pos[self.obj_b_sid]
        )
        self.tar_length = np.linalg.norm(
            self.sim.model.site_pos[self.tar_t_sid]
            - self.sim.model.site_pos[self.tar_b_sid]
        )

        self.goal_orient_range = goal_orient_range
        self.rsi = enable_rsi
        self.rsi_distance = rsi_distance

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        self.init_qpos[:-6] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.5  # place palm up

    def reset(self):
        # randomize target
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(
            low=self.goal_orient_range[0], high=self.goal_orient_range[1]
        )
        desired_orien[1] = self.np_random.uniform(
            low=self.goal_orient_range[0], high=self.goal_orient_range[1]
        )
        self.sim.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)

        if self.rsi:
            init_orien = np.zeros(3)
            init_orien[:2] = desired_orien[:2] + self.rsi_distance * (
                init_orien[:2] - desired_orien[:2]
            )
            self.sim.model.body_quat[self.obj_bid] = euler2quat(init_orien)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.robot.reset(qpos, qvel)

        return self.get_obs()
