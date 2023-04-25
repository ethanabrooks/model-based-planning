import datetime
import json
import os
from pprint import pprint
from typing import Optional

import torch
from rich.console import Console
from wandb.sdk.wandb_run import Run

import wandb
from utils.helpers import TAGS, project_name

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
        self._directory = run.dir

    @property
    def directory(self):
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

    def path(self, fname: str):
        return os.path.join(self.directory, fname)

    def save(self, path: str):
        wandb.save(path)


class DebugWriter(Writer):
    def __init__(self) -> None:
        timestamp = datetime.datetime.now().strftime("_%d:%m_%H:%M:%S")
        self._directory = os.path.join("/tmp", "restore-path", timestamp)
        os.makedirs(self._directory)

    @property
    def directory(self):
        return self._directory

    def dump_config(self, args: dict):
        del args

    def path(self, fname: str):
        del fname

    def save(self, path: str):
        del path

    def log(self, log: dict, step: int):
        pprint(log)
        print("step:", step)
