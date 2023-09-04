from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from environments.mujoco.rand_param_envs.base import RandomEnv
from environments.mujoco.rand_param_envs.gym import utils


class Walker2DRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0, **kwargs):
        self._max_episode_steps = 200
        self._elapsed_steps = -1  # the thing below takes one step
        RandomEnv.__init__(self, log_scale_limit, "walker2d.xml", 5, **kwargs)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        self._elapsed_steps += 1
        info = {"task": self.get_task(), "posbefore": posbefore, "posafter": posafter}
        if self._elapsed_steps == self._max_episode_steps:
            done = True
            info["bad_transition"] = True
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def _reset(self):
        ob = super()._reset()
        self._elapsed_steps = 0
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += 0.8
        self.viewer.cam.elevation = -20

    @classmethod
    def plot(
        cls,
        rollouts: np.ndarray,
        curr_task: np.ndarray,
        num_episodes: int = 1,
        image_path: Optional[str] = None,
    ):
        # plot the movement of the ant
        # print(pos)
        fig = plt.figure(figsize=(5, 4 * num_episodes))
        min_dim = -3.5
        max_dim = 3.5
        _ = max_dim - min_dim

        for i in range(num_episodes):
            plt.subplot(num_episodes, 1, i + 1)

            x = np.array([i["posbefore"] for _, _, _, _, i in rollouts[i]])
            plt.plot(x)

            plt.ylabel("position (ep {})".format(i), fontsize=15)

            if i == num_episodes - 1:
                plt.xlabel("time", fontsize=15)
                plt.ylabel("position (ep {})".format(i), fontsize=15)

        plt.tight_layout()
        if image_path is not None:
            # print(f"Saving plot to {image_path}")
            plt.savefig(image_path)
            plt.close()
        else:
            plt.show()

        return fig
