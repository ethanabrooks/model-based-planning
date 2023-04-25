import pdb

import numpy as np
import torch
from tqdm import tqdm

from trajectory.datasets import local
from trajectory.utils import discretization
from trajectory.utils.arrays import to_torch

from .d4rl import load_environment, qlearning_dataset_with_timeouts
from .preprocessing import dataset_preprocess_functions


def segment(observations, terminals, max_path_length, name: str):
    """
    segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    # Find indices of ones in Y
    indices, _ = np.where(terminals == 1)

    # Split X into segments
    trajectories = np.split(observations, indices + 1)

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

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
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=bool)
    for i, traj in enumerate(tqdm(trajectories, desc=f"Padding {name}")):
        path_length = path_lengths[i]
        trajectories_pad[i, :path_length] = traj
        early_termination[i, path_length:] = 1

    return trajectories_pad, early_termination, path_lengths


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env: str,
        sequence_length=250,
        step=10,
        discount=0.99,
        penalty=None,
        device="cuda:0",
        trajectory_transformer: bool = False,
    ):
        task_aware = local.is_task_aware(env)
        env = local.get_env_name(env)
        env = load_environment(env)
        name = env.spec.id
        self.sequence_length = sequence_length
        self.step = step
        self.device = device

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
        ##

        observations = dataset["observations"]
        actions = dataset["actions"]
        next_observations = dataset["next_observations"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        realterminals = dataset["realterminals"]
        if trajectory_transformer:
            terminals = realterminals

        def get_max_path_length(terms):
            ends, _ = np.where(terms)
            starts = np.pad(ends + 1, (1, 0))
            return np.max(np.diff(starts))

        self.max_path_length = max_path_length = get_max_path_length(terminals)

        print(
            f"[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}"
        )

        self.observations_raw = observations
        self.actions_raw = actions
        self.next_observations_raw = next_observations
        self.joined_raw = np.concatenate([observations, actions], axis=-1)
        self.rewards_raw = rewards
        self.terminals_raw = terminals

        ## terminal penalty
        if penalty is not None:
            terminal_mask = realterminals.squeeze()
            self.rewards_raw[terminal_mask] = penalty

        ## segment
        print("[ datasets/sequence ] Segmenting...", end=" ", flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(
            self.joined_raw, terminals, max_path_length, "observations/actions"
        )
        self.rewards_segmented, *_ = segment(
            self.rewards_raw, terminals, max_path_length, "rewards"
        )
        realterminals_segmented, *_ = segment(
            realterminals, terminals, max_path_length, "terminals"
        )
        realterminals_segmented, *_ = segment(
            realterminals, terminals, max_path_length, "realterminals"
        )
        print("✓")

        ## add missing final termination
        [last_term] = self.termination_flags[-1, :].nonzero()
        if last_term.size:
            [start_term, *_] = last_term
            realterminals_segmented[-1, start_term - 1] = True

        self.discount = discount
        self.discounts = (discount ** np.arange(max_path_length))[:, None]

        ## [ n_paths x max_path_length x 1 ]
        values_segmented = np.zeros(self.rewards_segmented.shape)
        max_ep_len = get_max_path_length(realterminals)
        exponents = np.triu(np.ones((max_ep_len, max_ep_len), dtype=int), 1).cumsum(
            axis=1
        )
        discount_array = np.triu(discount**exponents)

        rows, cols, zeros = np.where(realterminals_segmented)
        del zeros
        ep_ends = cols + 1
        ep_starts = np.pad(ep_ends, (1, 0))[:-1]
        row_change = np.diff(np.pad(rows, (1, 0))) > 0
        ep_starts[row_change] = 0

        for i, (row, start, end) in enumerate(
            tqdm(np.stack([rows, ep_starts, ep_ends], axis=1))
        ):
            assert start < end
            [ep_rewards] = self.rewards_segmented[row, start:end].T
            l = ep_rewards.size
            discounts = discount_array[:l, :l]
            ep_values = discounts @ ep_rewards
            values_segmented[row, start : end - 1] = ep_values[1:, None]

        self.values_segmented = values_segmented

        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]

        self.joined_raw = np.concatenate(
            [self.joined_raw, self.rewards_raw, self.values_raw], axis=-1
        )
        self.joined_segmented = np.concatenate(
            [self.joined_segmented, self.rewards_segmented, self.values_segmented],
            axis=-1,
        )

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(
            tqdm(self.path_lengths, desc="Assign indices")
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
        self.termination_flags = np.concatenate(
            [
                self.termination_flags,
                np.ones((n_trajectories, sequence_length - 1), dtype=bool),
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
        terminations = self.termination_flags[path_ind, start_ind : end_ind : self.step]
        terminations = np.pad(
            terminations,
            (0, self.sequence_length - len(terminations)),
            constant_values=True,
        )

        joined_discrete = self.discretizer.discretize(joined)
        joined, joined_discrete = [
            np.pad(x, [(0, self.sequence_length - len(joined_discrete)), (0, 0)])
            for x in [joined, joined_discrete]
        ]

        ## replace with termination token if the sequence has ended
        assert (
            joined[terminations] == 0
        ).all(), f"Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}"
        joined_discrete[terminations] = self.N

        ## [ (sequence_length / skip) x observation_dim]
        joined_discrete = to_torch(
            joined_discrete, device="cpu", dtype=torch.long
        ).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the episode boundary
        mask = torch.ones(joined_discrete.shape, dtype=torch.bool)
        mask[terminations] = False

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
