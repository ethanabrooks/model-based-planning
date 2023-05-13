import os
import re
from typing import Optional

import gym
import numpy as np
import wandb
import yaml
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
    artifact_names: list[str],
    task_aware: bool,
    ed: bool,
    truncate_episode: int,
) -> dict[str, np.ndarray]:
    buffers = []
    for i, artifact_name in enumerate(artifact_names, start=1):
        # download artifact
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(artifact_name)
        else:
            artifact = wandb.run.use_artifact(artifact_name)
        artifact_dir = artifact.download(root=os.getenv("WANDB_DIR"))

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
            desc=f"[ {i}/{len(artifact_names)} ] Restoring data from {artifact_dir}"
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
    rename = dict(state="observations", next_state="next_observations")

    def preprocess(v):
        v = v.numpy()
        b, *_, d = v.shape
        return v.reshape(b, d)

    dataset = {rename.get(k, k): preprocess(v) for k, v in memmap_tensors.items()}
    if task_aware:
        dataset["observations"] = add_task_to_obs(
            dataset["observations"], dataset["task"]
        )
    done_mdp = dataset["done_mdp"]
    (done_indices, _) = done_mdp.nonzero()
    episode_lengths = np.diff(done_indices)
    episode_length = np.unique(episode_lengths)
    try:
        (mask_size,) = episode_length
    except ValueError:
        raise RuntimeError(f"Episode length is not unique: {episode_length}")
    mask = np.zeros(mask_size, dtype=bool)
    done_bamdp_mask = np.zeros(mask_size, dtype=bool)
    terminations = np.zeros(mask_size, dtype=bool)
    mask[:truncate_episode] = 1
    done_bamdp_mask[-truncate_episode:] = 1
    terminations[truncate_episode - 1] = 1
    assert done_mdp.size % mask_size == 0, "Dataset size is not a multiple of mask size"
    tiles = int(done_mdp.size / mask_size)
    mask = np.tile(mask, tiles)
    done_bamdp_mask = np.tile(done_bamdp_mask, tiles)
    terminations = np.tile(terminations, tiles)
    dataset["done_mdp"][terminations] = 1
    dataset = {
        k: v[done_bamdp_mask if k == "done" else mask] for k, v in dataset.items()
    }
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
