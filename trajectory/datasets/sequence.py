import pdb
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn

import utils.helpers as utl
from trajectory.datasets import local
from trajectory.utils import discretization
from trajectory.utils.arrays import to_torch

from .d4rl import load_environment, qlearning_dataset_with_timeouts
from .preprocessing import dataset_preprocess_functions


def compute_episode_boundaries(done_tensor: np.ndarray):
    """
    Compute the start and end indices of episodes based on a given 'done' tensor.
    """
    ep_ends, _ = np.where(done_tensor)
    ep_ends += 1
    ep_starts = np.pad(ep_ends, (1, 0))[:-1]
    return ep_starts, ep_ends


def compute_rewards_for_tasks(observations: np.ndarray, done_tensor: np.ndarray, env):
    """
    Compute rewards based on tasks sampled for each episode.
    """
    ep_starts, ep_ends = compute_episode_boundaries(done_tensor)
    ep_tasks = env.sample_task(size=len(ep_starts))
    rewards = np.zeros_like(done_tensor)

    eps = np.concatenate([ep_starts[..., None], ep_ends[..., None], ep_tasks], axis=1)
    for start, end, *task in eps:
        assert start < end
        start = int(start)
        end = int(end)
        ep_obs = observations[start:end]
        task = np.array([task])
        ep_task_tile = np.tile(task, (end - start, 1))
        ep_rewards = env.get_reward(ep_obs, ep_task_tile)
        rewards[start:end] = ep_rewards[..., None]

    return rewards


def compute_values(
    rewards: np.ndarray, done_tensor: np.ndarray, discount_array: np.ndarray
):
    """
    Compute values based on rewards and given 'done' tensor.
    """
    ep_starts, ep_ends = compute_episode_boundaries(done_tensor)
    values = np.zeros_like(rewards)

    eps = np.concatenate([ep_starts[..., None], ep_ends[..., None]], axis=1)
    for start, end in eps:
        assert start < end
        ep_rewards = rewards[start:end].T[0]
        l = ep_rewards.size
        discounts = discount_array[:l, :l]
        ep_values = discounts @ ep_rewards
        values[start : end - 1] = ep_values[1:, None]

    return values


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env: str,
        trajectory_transformer: bool,
        sequence_length: int,
        step: int,
        discount: float,
        penalty: Optional[float],
        action_mask: bool,
    ):
        self.console = console = Console()
        self.sequence_length = sequence_length
        self.step = step
        self.action_mask = action_mask

        ed = local.is_ed(env)
        task_aware = local.is_task_aware(env)
        env = local.get_env_name(env)
        env = load_environment(env, test=True)
        utl.add_tag(
            f"{env.spec.max_episode_steps}-timesteps",
        )

        name = env.spec.id

        if local.get_artifact_name(name):
            dataset = local.load_dataset(
                env_name=name,
                task_aware=task_aware,
                ed=ed,
                truncate_episode=env.spec.max_episode_steps,
            )
        else:
            self.console.log("Loading...", end=" ")
            dataset = qlearning_dataset_with_timeouts(
                env.unwrapped, terminate_on_end=True
            )
            print("✓")

        preprocess_fn = dataset_preprocess_functions.get(name)
        if preprocess_fn:
            self.console.log("Modifying environment")
            dataset = preprocess_fn(dataset)

        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        done_bamdp = dataset["done"]
        done_mdp = dataset["done_mdp"]

        if trajectory_transformer:
            done_bamdp = done_mdp

        def get_max_path_length(terms):
            ends, _ = np.where(terms)
            starts = np.pad(ends + 1, (1, 0))
            return np.max(np.diff(starts))

        max_path_length = get_max_path_length(done_bamdp)

        console.log(
            f"Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}"
        )
        self.joined_raw = np.concatenate([observations, actions], axis=-1)

        ## done penalty
        if penalty is not None:
            done_mask = done_mdp.squeeze()
            rewards[done_mask] = penalty

        ## [ n_paths x max_path_length x 1 ]
        values = np.zeros(rewards.shape)
        max_ep_len = get_max_path_length(done_mdp)
        exponents = np.triu(np.ones((max_ep_len, max_ep_len), dtype=int), 1).cumsum(
            axis=1
        )
        discount_array = np.triu(discount**exponents)

        with Progress(
            TimeElapsedColumn(),
            *Progress.get_default_columns(),
        ) as progress:
            # Compute rewards based on BAMDP episodes
            rewards = compute_rewards_for_tasks(observations, done_bamdp, env)

            # Compute values based on MDP episodes
            values = compute_values(rewards, done_mdp, discount_array)

            ## segment
            progress.add_task("Segmenting...", total=None)

            def segment(x, name: str):
                """
                segment `observations` into trajectories according to `terminals`
                """
                assert len(x) == len(done_bamdp)
                observation_dim = x.shape[1]

                # Find indices of ones in Y
                indices, _ = np.where(done_bamdp == 1)

                # Split X into segments
                trajectories = np.split(x, indices + 1)

                if len(trajectories[-1]) == 0:
                    trajectories = trajectories[:-1]

                n_trajectories = len(trajectories)
                path_lengths = np.diff(np.pad(1 + indices, (1, 0)))

                if len(path_lengths) == len(trajectories) - 1:
                    path_lengths = np.append(path_lengths, len(trajectories[-1]))
                elif len(path_lengths) != len(trajectories):
                    raise ValueError(
                        f"Path lengths {path_lengths} and trajectories {trajectories} are not compatible"
                    )

                ## pad trajectories to be of equal length
                trajectories_pad = np.zeros(
                    (n_trajectories, max_path_length, observation_dim),
                    dtype=trajectories[0].dtype,
                )
                done_flags = np.zeros((n_trajectories, max_path_length), dtype=bool)
                for i, traj in enumerate(
                    progress.track(trajectories, description=f"Padding {name}")
                ):
                    path_length = path_lengths[i]
                    trajectories_pad[i, :path_length] = traj
                    done_flags[i, path_length:] = 1

                return trajectories_pad, done_flags, path_lengths

            self.joined_segmented, self.done_flags, self.path_lengths = segment(
                self.joined_raw, "observations/actions"
            )
            rewards_segmented, *_ = segment(rewards, "rewards")
            values_segmented, *_ = segment(values, "values")
            print("✓")

            ## add (r, V) to `joined`
            values_raw = values_segmented.squeeze(axis=-1).reshape(-1)
            values_mask = ~self.done_flags.reshape(-1)
            values_raw = values_raw[values_mask, None]

            ## get valid indices
            indices = []
            for path_ind, length in enumerate(
                progress.track(self.path_lengths - 1, description="Assign indices")
            ):
                starts = np.arange(1 - sequence_length, length)
                ends = starts + sequence_length
                idxs = path_ind * np.ones_like(starts)
                starts = np.clip(starts, 0, None)
                indices.append(np.stack([idxs, starts, ends]))

        self.rewards = rewards
        self.values = values_raw
        self.rewards_segmented = rewards_segmented
        self.values_segmented = values_segmented

        self.indices = np.concatenate(indices, axis=1).T
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.discount = discount
        self.joined_dim = (
            self.joined_raw.shape[1] + self.rewards.shape[1] + self.values.shape[1]
        )

    def __len__(self):
        return len(self.indices)


class DiscretizedDataset(SequenceDataset):
    def __init__(self, *args, N=50, discretizer="QuantileDiscretizer", **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        discretizer_class = getattr(discretization, discretizer)
        self.discretizer = discretizer_class(
            [self.joined_raw, self.rewards, self.values],
            N,
        )
        self.discretizer.discount = self.discount
        self.discretizer.observation_dim = self.observation_dim
        self.discretizer.action_dim = self.action_dim

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        obs_action = self.joined_segmented[path_ind, start_ind : end_ind : self.step]
        rewards = self.rewards_segmented[path_ind, start_ind : end_ind : self.step]
        values = self.values_segmented[path_ind, start_ind : end_ind : self.step]
        joined = np.concatenate([obs_action, rewards, values], axis=-1)
        done = self.done_flags[path_ind, start_ind : end_ind : self.step]
        done = np.pad(
            done,
            (0, self.sequence_length - len(done)),
            constant_values=True,
        )

        joined_discrete = self.discretizer.discretize(joined)
        joined, joined_discrete = [
            np.pad(x, [(0, self.sequence_length - len(joined_discrete)), (0, 0)])
            for x in [joined, joined_discrete]
        ]

        ## replace with done token if the sequence has ended
        assert (
            joined[done] == 0
        ).all(), (
            f"Everything after done, should be 0: {path_ind} | {start_ind} | {end_ind}"
        )
        joined_discrete[done] = self.N

        ## [ (sequence_length / skip) x observation_dim]
        joined_discrete = to_torch(
            joined_discrete, device="cpu", dtype=torch.long
        ).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the episode boundary
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        mask[done] = False
        if self.action_mask:
            mask[:, : self.observation_dim - 1] = False
            mask[:, self.observation_dim + self.action_dim - 1 :] = False

        ## flatten everything
        joined_discrete = joined_discrete.view(-1)
        mask = mask.view(-1)

        X = joined_discrete[:-1]
        Y = joined_discrete[1:]
        mask = mask[:-1]

        return X, Y, mask


class GoalDataset(DiscretizedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pdb.set_trace()

    def __getitem__(self, idx):
        X, Y, mask = super().__getitem__(idx)

        ## get path length for looking up the last transition in the trajcetory
        path_ind, start_ind, end_ind = self.indices[idx]
        path_length = self.path_lengths[path_ind]

        ## the goal is the first `observation_dim` dimensions of the last transition
        goal = self.joined_segmented[path_ind, path_length - 1, : self.observation_dim]
        goal_discrete = self.discretizer.discretize(
            goal, subslice=(0, self.observation_dim)
        )
        goal_discrete = (
            to_torch(goal_discrete, device="cpu", dtype=torch.long)
            .contiguous()
            .view(-1)
        )

        return X, goal_discrete, Y, mask
