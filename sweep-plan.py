from ray import tune

from plan import sweep
import torch

sweep(seed=tune.grid_search(list(range(torch.cuda.device_count()))))
