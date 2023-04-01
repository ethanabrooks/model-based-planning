import os
from typing import Optional
import re
import gym
import numpy as np
import yaml
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchsnapshot import Snapshot

from environments import parallel_envs
import environments  # noqa: F401

TASK_AWARE_PATTERN = re.compile(r"^TaskAware(.*)")


def get_env_name(env: str) -> str:
    matches = TASK_AWARE_PATTERN.match(env)
    if matches:
        return matches.group(1)
    return env


def is_task_aware(env: str) -> bool:
    return TASK_AWARE_PATTERN.match(env) is not None


def get_local_datasets():
    with open("local-datasets.yml") as f:
        local_datasets = yaml.load(f, Loader=yaml.FullLoader)
    return local_datasets


def get_data_path(env: str) -> Optional[str]:
    datasets = get_local_datasets()
    return datasets.get(env)


def is_local_dataset(env: str) -> bool:
    return get_data_path(env) is not None


def load_environment(env: str) -> gym.Env:
    thunk = parallel_envs.make_env(
        env,
        seed=None,
        rank=None,
        episodes_per_task=1,
        tasks=None,
        add_done_info=None,
    )
    env = thunk()
    env = NormalizedScoreWrapper(env)
    env = StateVectorWrapper(env)
    return env


def load_dataset(path: str, task_aware: bool) -> dict[str, np.ndarray]:
    full_path = os.path.join(os.environ["LOCAL_DATASET_PATH"], path)
    replay_buffer = ReplayBuffer(LazyMemmapStorage(0, scratch_dir="/tmp"))
    print(
        f"[ datasets/local ] Loading dataset from {full_path}...", end=" ", flush=True
    )
    snapshot = Snapshot(path=full_path)
    snapshot.restore(dict(replay_buffer=replay_buffer))
    print("✓")
    memmap_tensors = replay_buffer[: len(replay_buffer)]
    rename = dict(
        state="observations",
        next_state="next_observations",
        done="terminals",
        done_mdp="realterminals",
    )

    def preprocess(v):
        v = v.numpy()
        b, *_, d = v.shape
        return v.reshape(b, d)

    dataset = {rename.get(k, k): preprocess(v) for k, v in memmap_tensors.items()}
    if task_aware:
        dataset["observations"] = add_task_to_obs(
            dataset["observations"], dataset["task"]
        )
    return dataset


def add_task_to_obs(obs: np.ndarray, task: np.ndarray) -> np.ndarray:
    return np.concatenate([obs, task], axis=-1)


class TaskWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        try:
            [obs_size] = self.observation_space.shape
        except ValueError:
            raise RuntimeError(
                f"Observation space has shape {self.observation_space.shape}, but expected a 1D observation space."
            )
        try:
            [task_size] = self.get_task().shape
        except ValueError:
            raise RuntimeError(
                f"Task has shape {self.get_task().shape}, but expected a 1D task."
            )

        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=[obs_size + task_size],
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return add_task_to_obs(observation, self.get_task())


class NormalizedScoreWrapper(gym.Wrapper):
    def get_normalized_score(self, score):
        return score  # TODO


class StateVectorWrapper(gym.Wrapper):
    def state_vector(self):
        return self.env.unwrapped._get_obs()
