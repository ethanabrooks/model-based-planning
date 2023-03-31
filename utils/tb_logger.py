import datetime
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_full_output_folder(args, output_name: str = None):
    if output_name is None:
        output_name = (
            str(args.seed) + "_" + datetime.datetime.now().strftime("_%d:%m_%H:%M:%S")
        )
    dir_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
    )
    dir_path = os.path.join(dir_path, "logs")

    return os.path.join(
        os.path.join(dir_path, "logs_{}".format(args.dataset)),
        output_name,
    )


class TBLogger:
    def __init__(self, args, output_name: str = None):
        self.full_output_folder = get_full_output_folder(args, output_name)
        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)
        with open(os.path.join(self.full_output_folder, "config.json"), "w") as f:
            config = {
                k: v
                for k, v in args.as_dict().items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            config.update(device=device.type)
            json.dump(config, f, indent=2)

        self.writer = SummaryWriter(log_dir=self.full_output_folder)
        print("logging under", self.full_output_folder)

    def add(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)
