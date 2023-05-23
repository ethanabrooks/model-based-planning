import datetime
import os
import re
from typing import Callable, Optional

import gym
import numpy as np
import requests
import wandb
import yaml
from braceexpand import braceexpand
from rich.console import Console
from rich.progress import track
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchsnapshot import Snapshot

import environments  # noqa: F401
from environments import parallel_envs
from utils.helpers import tmp_dir
from utils.timer import Timer

TASK_AWARE_PATTERN = re.compile(r"^TaskAware(.*)")
ED_PATTERN = re.compile(r"^ED(.*)")
console = Console()


def get_env_name(env: str) -> str:
    matches = TASK_AWARE_PATTERN.match(env)
    if matches:
        return matches.group(1)
    matches = ED_PATTERN.match(env)
    if matches:
        return matches.group(1)
    return env


def is_task_aware(env: str) -> bool:
    return TASK_AWARE_PATTERN.match(env) is not None


def is_ed(env: str) -> bool:
    return ED_PATTERN.match(env) is not None


with open("local-datasets.yml") as f:
    LOCAL_DATASETS = yaml.load(f, Loader=yaml.FullLoader)


def get_local_datasets():
    return LOCAL_DATASETS


def get_artifact_name(env: str) -> Optional[list[str]]:
    datasets = get_local_datasets()
    return datasets.get(env)


def is_local_dataset(env: str) -> bool:
    return get_artifact_name(env) is not None


def load_environment(env: str) -> gym.Env:
    thunk = parallel_envs.make_env(
        env,
        seed=None,
        rank=None,
        episodes_per_task=None,
        tasks=None,
        add_done_info=None,
    )
    env = thunk()
    env = NormalizedScoreWrapper(env)
    env = StateVectorWrapper(env)
    env = GetObsWrapper(env)
    return env


def load_dataset(
    env_name: str,
    task_aware: bool,
    ed: bool,
    truncate_episode: int,
    train_task_mask: Optional[Callable[[np.ndarray], np.ndarray]],
) -> dict[str, np.ndarray]:
    buffers = []
    artifact_path = get_artifact_name(env_name)
    if not isinstance(artifact_path, list):
        artifact_path = list(braceexpand(artifact_path))
    for i, path in enumerate(artifact_path, start=1):
        if os.path.exists(path):
            artifact_dir = path
        else:
            # download artifact
            if wandb.run is None:
                api = wandb.Api()
                artifact = api.artifact(path)
            else:
                artifact = wandb.run.use_artifact(path)
            while True:
                try:
                    artifact_dir = artifact.download(
                        root=os.path.join(
                            os.getenv("WANDB_DIR"),
                            env_name,
                            datetime.datetime.now().strftime("_%d:%m_%H:%M:%S"),
                        )
                    )
                    break
                except requests.exceptions.ChunkedEncodingError as e:
                    console.log(e)
                    console.log("\nRetrying download...\n")

        # load buffers
        run_buffers = {}
        for path in os.listdir(f"{artifact_dir}/0"):
            if re.match(r"\d+", path):
                replay_buffer = ReplayBuffer(
                    LazyMemmapStorage(0, scratch_dir=tmp_dir())
                )
                run_buffers[path] = replay_buffer
        snapshot = Snapshot(path=artifact_dir)
        with Timer(
            desc=f"[ {i}/{len(artifact_path)} ] Restoring data from {artifact_dir}"
        ):
            snapshot.restore(run_buffers)
        buffers.extend(run_buffers.values())

    # merge buffers
    size = sum(len(buffer) for buffer in buffers)
    replay_buffer = ReplayBuffer(LazyMemmapStorage(size, scratch_dir=tmp_dir()))
    for buffer in track(buffers, description="Merging buffers"):
        tensordict = buffer[:]
        [done_mdp] = tensordict["done_mdp"].T
        nonzero = (*_, last) = done_mdp.nonzero()
        last = last.item()
        tensordict.set_at_("done", True, last)  # terminate last transition per task
        tensordict = tensordict[: last + 1]  # eliminate partial episodes
        if ed:
            cutoff = nonzero[nonzero.numel() // 2]
            tensordict = tensordict[cutoff:]
        replay_buffer.extend(tensordict)

    memmap_tensors = replay_buffer[: len(replay_buffer)]
    if train_task_mask is not None:
        train_task_mask = train_task_mask(memmap_tensors["task"])
        memmap_tensors = memmap_tensors[train_task_mask]
    [done_mdp] = memmap_tensors["done_mdp"].T
    [done_indices] = done_mdp.nonzero().T
    episode_lengths = np.diff(np.concatenate([np.array([0]), 1 + done_indices]))
    episode_length = np.unique(episode_lengths)
    try:
        [mask_size] = episode_length  # single episode length: can vectorize computation
    except ValueError:
        console.log("Episode length is not unique:")
        console.log(episode_length)
        mask_size = None  # computation must be performed iteratively
    if mask_size is None:
        replay_buffer = ReplayBuffer(LazyMemmapStorage(size, scratch_dir=tmp_dir()))
        for length in track(episode_lengths, description="Truncating episodes"):
            episode = memmap_tensors[:length]
            memmap_tensors = memmap_tensors[length:]
            assert episode["done_mdp"][-1], f"Episode is not terminated: {episode}"
            episode = episode[:truncate_episode]
            replay_buffer.extend(episode)
        assert len(memmap_tensors) == 0
        dataset = replay_buffer[:]

    else:
        mask = np.zeros(mask_size, dtype=bool)
        done_bamdp_mask = np.zeros(mask_size, dtype=bool)
        terminations = np.zeros(mask_size, dtype=bool)
        mask[:truncate_episode] = 1
        done_bamdp_mask[-truncate_episode:] = 1
        terminations[truncate_episode - 1] = 1
        assert (
            done_mdp.size % mask_size == 0
        ), "Dataset size is not a multiple of mask size"
        tiles = int(done_mdp.size / mask_size)
        mask = np.tile(mask, tiles)
        done_bamdp_mask = np.tile(done_bamdp_mask, tiles)
        terminations = np.tile(terminations, tiles)
        dataset["done_mdp"][terminations] = 1
        dataset = {
            k: v[done_bamdp_mask if k == "done" else mask] for k, v in dataset.items()
        }
    if task_aware:
        dataset["observations"] = add_task_to_obs(
            dataset["observations"], dataset["task"]
        )
    rename = dict(state="observations", next_state="next_observations")

    def preprocess(v):
        v = v.numpy()
        b, *_, d = v.shape
        return v.reshape(b, d)

    return {rename.get(k, k): preprocess(v) for k, v in dataset.items()}


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

    def _get_obs(self):
        obs = self.env._get_obs()
        return self.observation(obs)


class NormalizedScoreWrapper(gym.Wrapper):
    def get_normalized_score(self, score):
        return score  # TODO


class StateVectorWrapper(gym.Wrapper):
    def state_vector(self):
        return self.env.unwrapped._get_obs()


class GetObsWrapper(gym.Wrapper):
    def _get_obs(self):
        return self.env.unwrapped._get_obs()
