import importlib
import os
import random
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from tap import Tap

from .git_utils import get_git_rev, save_git_diff

console = Console()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            exp_name.append(f"{label}{val}")
        exp_name = "_".join(exp_name)
        exp_name = exp_name.replace("/_", "/")
        return exp_name

    return _fn


class Parser(Tap):
    def save(self):
        fullpath = os.path.join(self.savepath, "args.json")
        console.log(f"Saved args to {fullpath}")
        super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, args: Optional[list[str]] = None, experiment=None):
        args = super().parse_args(args=args, known_only=True)
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, "config"):
            return args
        args = self.read_config(args, experiment)
        self.add_extras(args)
        self.set_seed(args)
        self.get_commit(args)
        self.generate_exp_name(args)
        self.save_diff(args)
        return args

    def read_config(self, args, experiment):
        """
        Load parameters from config file
        """
        dataset = args.dataset.replace("-", "_")
        console.log(f"Reading config: {args.config}:{dataset}")
        module = importlib.import_module(args.config)
        params = getattr(module, "base")[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            console.log(f"Using overrides: {dict(config=args.config, dataset=dataset)}")
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            console.log(
                f"Not using overrides: {dict(config=args.config, dataset=args.dataset)}"
            )

        for key, val in params.items():
            setattr(args, key, val)

        return args

    def add_extras(self, args):
        """
        Override config parameters with command-line arguments
        """
        extras = args.extra_args
        if not len(extras):
            return

        console.log(f"Found extras: {extras}")
        assert (
            len(extras) % 2 == 0
        ), f"Found odd number ({len(extras)}) of extras: {extras}"
        for i in range(0, len(extras), 2):
            key = extras[i].replace("--", "")
            val = extras[i + 1]
            assert hasattr(args, key), f"{key} not found in config: {args.config}"
            old_val = getattr(args, key)
            old_type = type(old_val)
            console.log(f"Overriding config | {key} : {old_val} --> {val}")
            if val == "None":
                val = None
            elif val == "latest":
                val = "latest"
            elif old_type in [bool, type(None)]:
                val = eval(val)
            else:
                val = old_type(val)
            setattr(args, key, val)

    def set_seed(self, args):
        console.log(f"Seed: {getattr(args, 'seed', None)}")
        if "seed" not in dir(args):
            return
        set_seed(args.seed)

    def generate_exp_name(self, args):
        if "exp_name" not in dir(args):
            return
        exp_name = getattr(args, "exp_name")
        if callable(exp_name):
            exp_name_string = exp_name(args)
            console.log(f"Setting exp_name to: {exp_name_string}")
            setattr(args, "exp_name", exp_name_string)

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, "diff.txt"))
        except:
            console.log("WARNING: did not save git diff")
