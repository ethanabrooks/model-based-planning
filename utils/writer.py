import datetime
import json
import os
import pickle
from pathlib import Path
from pprint import pprint
from typing import Optional

import torch
import wandb
from rich.console import Console
from wandb.sdk.wandb_run import Run

from utils.helpers import TAGS, project_name, tmp_dir

console = Console()


class Writer:
    def __init__(
        self, config: dict, dataset: str, name: str, notes: str, run: Optional[Run]
    ) -> None:
        name = f"{name}-{dataset}"
        if run is None:
            wandb.init(
                config=config, name=name, notes=notes, project=project_name(), tags=TAGS
            )
            run = wandb.run
        self.run = run
        assert wandb.run is not None, "Should be using DebugWriter is not using wandb."
        self._directory = tmp_dir()

    @property
    def directory(self) -> Path:
        return self._directory

    def load_artifact(self, name: str):
        return self.run.use_artifact(name).download()

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def dump_config(self, args: dict):
        with open(os.path.join(self.directory, "config.json"), "w") as f:
            config = {
                k: v
                for k, v in args.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            config.update(device=device.type)
            json.dump(config, f, indent=2)

    def dump_pickle(self, obj, fname: str):
        save_path = self.path(fname)
        with save_path.open("wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, fname: str):
        load_path = self.path(fname)
        with load_path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def make(
        debug: bool,
        config: dict,
        dataset: str,
        name: str,
        notes: str,
        run: Optional[Run],
    ):
        return (
            DebugWriter()
            if debug
            else Writer(config=config, dataset=dataset, name=name, notes=notes, run=run)
        )

    def path(self, fname: str) -> Path:
        return Path(self.directory, fname)

    def save(self, path: Path):
        wandb.save(str(path))


class DebugWriter(Writer):
    def __init__(self) -> None:
        timestamp = datetime.datetime.now().strftime("_%d:%m_%H:%M:%S")
        self._directory = Path("/tmp", "restore-path", timestamp)
        self._directory.mkdir(parents=True)

    @property
    def directory(self) -> Path:
        return self._directory

    def dump_config(self, args: dict):
        del args

    def save(self, path: str):
        del path

    def log(self, log: dict, step: int):
        pprint(log)
        print("step:", step)
