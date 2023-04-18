from ray import tune

from plan import sweep

sweep(seed=tune.grid_search(list(range(8))))
