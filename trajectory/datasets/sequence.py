import pdb
from typing import Optional

import numpy as np
import torch
from rich.progress import track

from trajectory.datasets import local
from trajectory.utils import discretization
from trajectory.utils.arrays import to_torch

from .d4rl import load_environment, qlearning_dataset_with_timeouts
from .preprocessing import dataset_preprocess_functions


def segment(observations, done, max_path_length, name: str):
    """
    segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(done)
    observation_dim = observations.shape[1]

    # Find indices of ones in Y
    indices, _ = np.where(done == 1)

    # Split X into segments
    trajectories = np.split(observations, indices + 1)

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
        (n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype
    )
    done_flags = np.zeros((n_trajectories, max_path_length), dtype=bool)
    for i, traj in enumerate(track(trajectories, description=f"Padding {name}")):
        path_length = path_lengths[i]
        trajectories_pad[i, :path_length] = traj
        done_flags[i, path_length:] = 1

    return trajectories_pad, done_flags, path_lengths


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env: str,
        trajectory_transformer: bool,
        sequence_length: int,
        step: int,
        discount: float,
        penalty: Optional[float],
    ):
        task_aware = local.is_task_aware(env)
        env = local.get_env_name(env)
        env = load_environment(env)
        name = env.spec.id
        self.sequence_length = sequence_length
        self.step = step

        artifact_names = local.get_artifact_name(name)
        if artifact_names:
            dataset = local.load_dataset(artifact_names, task_aware)
        else:
            print("[ datasets/sequence ] Loading...", end=" ", flush=True)
            dataset = qlearning_dataset_with_timeouts(
                env.unwrapped, terminate_on_end=True
            )
            print("✓")

        preprocess_fn = dataset_preprocess_functions.get(name)
        if preprocess_fn:
            print("[ datasets/sequence ] Modifying environment")
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

        print(
            f"[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}"
        )
        self.joined_raw = np.concatenate([observations, actions], axis=-1)

        ## done penalty
        if penalty is not None:
            done_mask = done_mdp.squeeze()
            rewards[done_mask] = penalty

        ## segment
        print("[ datasets/sequence ] Segmenting...", end=" ", flush=True)
        self.joined_segmented, self.done_flags, self.path_lengths = segment(
            self.joined_raw, done_bamdp, max_path_length, "observations/actions"
        )
        rewards_segmented, *_ = segment(rewards, done_bamdp, max_path_length, "rewards")
        done_segmented, *_ = segment(done_mdp, done_bamdp, max_path_length, "done_mdp")
        print("✓")

        ## add missing final termination
        [last_term] = self.done_flags[-1, :].nonzero()
        if last_term.size:
            [start_term, *_] = last_term
            done_segmented[-1, start_term - 1] = True

        ## [ n_paths x max_path_length x 1 ]
        values_segmented = np.zeros(rewards_segmented.shape)
        max_ep_len = get_max_path_length(done_mdp)
        exponents = np.triu(np.ones((max_ep_len, max_ep_len), dtype=int), 1).cumsum(
            axis=1
        )
        discount_array = np.triu(discount**exponents)

        rows, cols, zeros = np.where(done_segmented)
        del zeros
        ep_ends = cols + 1
        ep_starts = np.pad(ep_ends, (1, 0))[:-1]
        row_change = np.diff(np.pad(rows, (1, 0))) > 0
        ep_starts[row_change] = 0

        for i, (row, start, end) in enumerate(
            track(np.stack([rows, ep_starts, ep_ends], axis=1))
        ):
            assert start < end
            [ep_rewards] = rewards_segmented[row, start:end].T
            l = ep_rewards.size
            discounts = discount_array[:l, :l]
            ep_values = discounts @ ep_rewards
            values_segmented[row, start : end - 1] = ep_values[1:, None]

        ## add (r, V) to `joined`
        values_raw = values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.done_flags.reshape(-1)
        values_raw = values_raw[values_mask, None]

        self.joined_raw = np.concatenate(
            [self.joined_raw, rewards, values_raw], axis=-1
        )
        self.joined_segmented = np.concatenate(
            [self.joined_segmented, rewards_segmented, values_segmented],
            axis=-1,
        )

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(
            track(self.path_lengths, description="Assign indices")
        ):
            end = length - 1
            for j in range(1, sequence_length):
                indices.append((path_ind, 0, j))  # train prefixes
            for i in range(end):
                indices.append((path_ind, i, i + sequence_length))

        self.indices = np.array(indices)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.joined_segmented = np.concatenate(
            [
                self.joined_segmented,
                np.zeros((n_trajectories, sequence_length - 1, joined_dim)),
            ],
            axis=1,
        )

    def __len__(self):
        return len(self.indices)


class DiscretizedDataset(SequenceDataset):
    def __init__(self, *args, N=50, discretizer="QuantileDiscretizer", **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        discretizer_class = getattr(discretization, discretizer)
        self.discretizer = discretizer_class(self.joined_raw, N)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind : end_ind : self.step]
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
