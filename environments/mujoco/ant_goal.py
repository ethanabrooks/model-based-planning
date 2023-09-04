import random
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from environments.mujoco.ant import AntEnv


class AntGoalEnv(AntEnv):
    def __init__(
        self,
        max_episode_steps=200,
        test_threshold: Optional[float] = None,
        test: bool = False,
    ):
        self.test_threshold = test_threshold
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        super(AntGoalEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))
        goal_vector = self.goal_pos - torso_xyz_before[:2]
        goal_distance = np.linalg.norm(goal_vector)
        direct = goal_vector / goal_distance

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        torso_velocity = torso_velocity[:2]
        torso_velocity_norm = np.linalg.norm(torso_velocity)
        if goal_distance < 0.5:
            forward_reward = 1
        else:
            forward_reward = np.dot((torso_velocity / torso_velocity_norm), direct)
            assert -1 <= forward_reward <= 1

        ctrl_cost = 0.5 * np.square(action).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        distance_traveled = np.linalg.norm(torso_xyz_before)
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                distance_traveled=distance_traveled,
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                torso_velocity=torso_velocity_norm,
                task=self.get_task(),
            ),
        )

    def sample_tasks(self, num_tasks):
        a = np.array([random.random() for _ in range(num_tasks)]) * 2 * np.pi
        r = 2 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        if self.test_threshold:
            r = self.test_threshold * np.ones_like(r)
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return np.array(self.goal_pos)

    def plot_task(curr_task: np.ndarray):
        plt.plot(curr_task[0], curr_task[1], "rx")

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )[:30]


class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
                self.goal_pos,
            ]
        )
