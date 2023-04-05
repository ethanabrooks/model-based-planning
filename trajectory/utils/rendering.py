import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import mujoco_py as mjc
import pdb
from typing import Optional

import wandb

from trajectory.datasets.local import TaskWrapper

from .arrays import to_np
from .video import save_video, save_videos
from ..datasets import load_environment, get_preprocess_fn


def make_renderer(dataset, renderer, env: gym.Env, **_):
    render_class = getattr(sys.modules[__name__], renderer)
    ## get dimensions in case the observations are preprocessed
    preprocess_fn = get_preprocess_fn(dataset)
    observation = env.reset()
    observation = preprocess_fn(observation)
    return render_class(env=env, observation_dim=observation.size)


def split(sequence, observation_dim, action_dim):
    assert sequence.shape[1] == observation_dim + action_dim + 2
    observations = sequence[:, :observation_dim]
    actions = sequence[:, observation_dim : observation_dim + action_dim]
    rewards = sequence[:, -2]
    values = sequence[:, -1]
    return observations, actions, rewards, values


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    qstate_dim = qpos_dim + qvel_dim
    name = env.spec.id

    if "ant" in name:
        ypos = np.zeros(1)
        state = np.concatenate([ypos, state])

    if state.size == qpos_dim - 1 or state.size == qstate_dim - 1:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])

    if state.size == qpos_dim:
        qvel = np.zeros(qvel_dim)
        state = np.concatenate([state, qvel])

    if "ant" in name and state.size > qpos_dim + qvel_dim:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])[:qstate_dim]

    if "HalfCheetahVel" in name or "HalfCheetahDir" in name or "AntGoal" in name:
        state = state[: qpos_dim + qvel_dim]

    assert state.size == qpos_dim + qvel_dim

    env.set_state(state[:qpos_dim], state[qpos_dim:])


def rollout_from_state(env, state, actions):
    set_state(env, state)
    observations = [env._get_obs()]

    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions) + 1):
        ## if terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations)


class DebugRenderer:
    def __init__(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        return np.zeros((10, 10, 3))

    def render_plan(self, *args, **kwargs):
        pass

    def render_rollout(self, *args, **kwargs):
        pass


class Renderer:
    def __init__(self, env, observation_dim=None, action_dim=None):
        if type(env) is str:
            self.env = load_environment(env)
        else:
            self.env = env

        self.observation_dim = observation_dim or np.prod(
            self.env.observation_space.shape
        )
        self.action_dim = action_dim or np.prod(self.env.action_space.shape)
        try:
            sim = self.env.sim
        except AttributeError:
            sim = None
        self.viewer = (
            None if sim is None else mjc.MjRenderContextOffscreen(self.env.sim)
        )

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

    def render(self, observation, dim=256, render_kwargs=None):
        if self.viewer is None:
            return
        observation = to_np(observation)

        if render_kwargs is None:
            render_kwargs = {
                "trackbodyid": 2,
                "distance": 3,
                "lookat": [0, -0.5, 1],
                "elevation": -20,
            }

        for key, val in render_kwargs.items():
            if key == "lookat":
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        set_state(self.env, observation)

        if type(dim) == int:
            dim = (dim, dim)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def render_plan(self, savepath, sequence, state, fps=30):
        """
        state : np.array[ observation_dim ]
        sequence : np.array[ horizon x transition_dim ]
            as usual, sequence is ordered as [ s_t, a_t, r_t, V_t, ... ]
        """

        if len(sequence) == 1:
            return

        sequence = to_np(sequence)

        ## compare to ground truth rollout using actions from sequence
        actions = sequence[
            :-1, self.observation_dim : self.observation_dim + self.action_dim
        ]
        rollout_states = rollout_from_state(self.env, state, actions)

        videos = [
            self.renders(sequence[:, : self.observation_dim]),
            self.renders(rollout_states),
        ]

        save_videos(savepath, *videos, fps=fps)

    def render_rollout(self, savepath, states, **video_kwargs):
        images = self(states)
        save_video(savepath, images, **video_kwargs)


class PointRenderer:
    def __init__(self, env: gym.Env, observation_dim: int):
        self.env = env
        self.observation_dim = observation_dim
        self.action_dim = np.prod(self.env.action_space.shape)

    def renders(
        self, savepath: str, X: np.ndarray, actions: Optional[np.ndarray] = None
    ):
        fig, ax = plt.subplots()

        assert X.ndim == 2
        if isinstance(self.env, TaskWrapper):
            states, tasks = np.split(X, 2, -1)
            task, *_ = tasks
            assert np.all(tasks == task[None])

            # plot the task using * notation to unpack the task array
            ax.plot(*task, "r*")
        else:
            states = X

        # plot the states
        ax.plot(*states.T, "-o")

        # plot the actions as arrows
        if actions is None:
            actions = np.diff(states, axis=0)

        for state, action in zip(states, actions):
            ax.arrow(*state, *action)

        plt.savefig(savepath + ".png")
        if wandb.run is not None:
            wandb.log({savepath: wandb.Image(fig)})
        plt.close()
        print(f"[ attentive/utils/visualization ] Saved to: {savepath}")

    def render_plan(self, savepath, sequence, state):
        """
        state : np.array[ observation_dim ]
        sequence : np.array[ horizon x transition_dim ]
            as usual, sequence is ordered as [ s_t, a_t, r_t, V_t, ... ]
        """
        del state

        if len(sequence) == 1:
            # raise RuntimeError(f'horizon is 1 in Renderer:render_plan: {sequence.shape}')
            return

        sequence = to_np(sequence)

        states, actions, *_ = split(sequence, self.observation_dim, self.action_dim)
        self.renders(savepath, states, actions)

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list:
            states = np.stack(states, axis=0)
        self.renders(savepath, states)


# --------------------------------- planning callbacks ---------------------------------#
