import glob
import json
import os
import pdb
import pickle
import sys
import time

import torch


def mkdir(savepath, prune_fname=False):
    """
    returns `True` iff `savepath` is created
    """
    if prune_fname:
        savepath = os.path.dirname(savepath)
    if not os.path.exists(savepath):
        try:
            os.makedirs(savepath)
        except:
            print(
                f"[ utils/serialization ] Warning: did not make directory: {savepath}"
            )
            return False
        return True
    else:
        return False


def get_latest_epoch(loadpath):
    states = glob.glob1(loadpath, "state_*")
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace("state_", "").replace(".pt", ""))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch


def load_model(*loadpath, epoch=None, device="cuda:0"):
    loadpath = os.path.join(*loadpath)
    config_path = os.path.join(loadpath, "model_config.pkl")

    if epoch == "latest":
        epoch = get_latest_epoch(loadpath)

    print(f"[ utils/serialization ] Loading model epoch: {epoch}")
    state_path = os.path.join(loadpath, f"state_{epoch}.pt")

    config = pickle.load(open(config_path, "rb"))
    state = torch.load(state_path)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model = config()
    model.to(device)
    model.load_state_dict(state, strict=True)

    print(f"\n[ utils/serialization ] Loaded config from {config_path}\n")
    print(config)

    return model, epoch


def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, "rb"))
    print(f"[ utils/serialization ] Loaded config from {loadpath}")
    print(config)
    return config


def load_from_config(*loadpath):
    config = load_config(*loadpath)
    return config.make()


def load_args(*loadpath):
    from .setup import Parser

    loadpath = os.path.join(*loadpath)
    args_path = os.path.join(loadpath, "args.json")
    args = Parser()
    args.load(args_path)
    return args
