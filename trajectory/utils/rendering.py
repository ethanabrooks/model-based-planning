import sys

import gym
import matplotlib.pyplot as plt
import mujoco_py as mjc
import numpy as np

import wandb

from ..datasets import get_preprocess_fn, load_environment
from .arrays import to_np
from .video import save_video, save_videos


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

        self.viewer.render(*dim, camera_id=-1)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def render_plan(self, savepath, sequence, state, env, fps=30):
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
            self.renders(sequence[:, : self.observation_dim], env),
            self.renders(rollout_states, env),
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

    def renders(self, savepath: str, states: np.ndarray, env):
        figsize = (5.5, 4)
        fig, axis = plt.subplots(1, 1, figsize=figsize)
        xlim = (-1.3, 1.3)

        ## Necessary because of insane circular import error
        from environments.navigation.point_robot import semi_circle_goal_sampler

        if env.unwrapped.goal_sampler == semi_circle_goal_sampler:
            ylim = (-0.3, 1.3)
        else:
            ylim = (-1.3, 1.3)
        curr_task = env.get_task()

        # plot goal
        axis.scatter(*curr_task, marker="x", color="k", s=50)
        # radius where we get reward
        if hasattr(env, "goal_radius"):
            circle1 = plt.Circle(
                curr_task, env.goal_radius, color="c", alpha=0.2, edgecolor="none"
            )
            plt.gca().add_artist(circle1)

        # plot (semi-)circle
        r = 1.0
        if env.unwrapped.goal_sampler == semi_circle_goal_sampler:
            angle = np.linspace(0, np.pi, 100)
        else:
            angle = np.linspace(0, 2 * np.pi, 100)
        goal_range = r * np.array((np.cos(angle), np.sin(angle)))
        plt.plot(goal_range[0], goal_range[1], "k--", alpha=0.1)

        # plot trajectory
        axis.plot(states[:, 0], states[:, 1], "-")
        axis.scatter(*states[0, :2], marker=".", s=50)

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        plt.savefig(savepath + ".png")
        if wandb.run is not None:
            wandb.log({savepath: wandb.Image(fig)})
        plt.close()
        print(f"[ attentive/utils/visualization ] Saved to: {savepath}")

    def render_plan(self, savepath, sequence, state, env):
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
        self.renders(savepath, states, env)

    def render_rollout(self, savepath, states, env, **video_kwargs):
        if type(states) is list:
            states = np.stack(states, axis=0)
        self.renders(savepath, states, env)


# --------------------------------- planning callbacks ---------------------------------#
