from argparse import ArgumentParser

from ray import tune

from plan import sweep

parser = ArgumentParser()
parser.add_argument("--gpus-per-proc", type=int, default=1)
parser.add_argument("--seeds", type=int, default=8)
args, rest_args = parser.parse_known_args()
sweep(
    args=rest_args,
    config=dict(seed=tune.grid_search(list(range(args.seeds)))),
    gpus_per_proc=args.gpus_per_proc,
)
