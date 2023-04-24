from trajectory.utils import watch

# ------------------------ base ------------------------#

logbase = "logs/"
gpt_expname = "gpt/azure"

## automatically make experiment names for planning
## by labelling folders with these args
args_to_watch = [
    ("prefix", ""),
    ("plan_freq", "freq"),
    ("horizon", "H"),
    ("beam_width", "beam"),
]

base = {
    "train": {
        "N": 100,
        "discount": 0.99,
        "n_layer": 4,
        "n_head": 4,
        ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
        "n_epochs_ref": 10,
        "n_saves": 3,
        "logbase": logbase,
        "device": "cuda",
        "n_embd": 32,
        "batch_size": 256,
        "learning_rate": 6e-4,
        "lr_decay": True,
        "seed": 42,
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "step": 1,
        "subsampled_sequence_length": 10,
        "termination_penalty": -100,
        "exp_name": gpt_expname,
        "discretizer": "QuantileDiscretizer",
        "action_weight": 5,
        "reward_weight": 1,
        "value_weight": 1,
    },
    "plan": {
        "logbase": logbase,
        "gpt_loadpath": gpt_expname,
        "gpt_epoch": "latest",
        "device": "cuda",
        "renderer": "Renderer",
        "plan_freq": 1,
        "horizon": 15,
        "beam_width": 128,
        "n_expand": 2,
        "k_obs": 1,
        "k_act": None,
        "cdf_obs": None,
        "cdf_act": 0.6,
        "percentile": "mean",
        "max_context_transitions": 5,
        "prefix_context": True,
        "vis_freq": 50,
        "exp_name": watch(args_to_watch),
        "prefix": "plans/defaults/",
        "suffix": "0",
        "verbose": True,
    },
}

# ------------------------ locomotion ------------------------#

## for all halfcheetah environments, you can reduce the planning horizon and beam width without
## affecting performance. good for speed and sanity.

halfcheetah_medium_v2 = halfcheetah_medium_replay_v2 = {
    "plan": {
        "horizon": 5,
        "beam_width": 32,
        "loadpath": "rldl/In-Context Model-Based Planning/rfkmp3sq",
    }
}

halfcheetah_medium_expert_v2 = {
    "plan": {
        "beam_width": 32,
    },
}

## if you leave the dictionary empty, it will use the base parameters
hopper_medium_expert_v2 = hopper_medium_v2 = walker2d_medium_v2 = {}

## hopper and wlaker2d are a little more sensitive to planning hyperparameters;
## proceed with caution when reducing the horizon or increasing the planning frequency

hopper_medium_replay_v2 = {
    "train": {
        ## train on the medium-replay datasets longer
        "n_epochs_ref": 80,
    },
}

walker2d_medium_expert_v2 = {
    "plan": {
        ## also safe to reduce the horizon here
        "horizon": 5,
    },
}

walker2d_medium_replay_v2 = {
    "train": {
        ## train on the medium-replay datasets longer
        "n_epochs_ref": 80,
    },
    "plan": {
        ## can reduce beam width, but need to adjust action sampling
        ## distribution and increase horizon to accomodate
        "horizon": 20,
        "beam_width": 32,
        "k_act": 40,
        "cdf_act": None,
    },
}

ant_medium_v2 = ant_medium_replay_v2 = ant_random_v2 = {
    "train": {
        ## reduce batch size because the dimensionality is larger
        "batch_size": 128,
    },
    "plan": {
        "horizon": 5,
    },
}

halfcheetah_medium_v2 = {
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/rfkmp3sq",
    },
}

SparsePointEnv_v0 = {
    "plan": {
        "renderer": "PointRenderer",
        "loadpath": "rldl/In-Context Model-Based Planning/cxuuozux",
    },
}

TaskAwareSparsePointEnv_v0 = {
    "plan": {
        "renderer": "PointRenderer",
        "loadpath": "rldl/In-Context Model-Based Planning/42saqsl0",
    },
}

HalfCheetahVel_v0 = {
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/vw1i5wf5",
    },
}

TaskAwareHalfCheetahVel_v0 = {
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/q72fbi9c",
    },
}

HalfCheetahDir_v0 = {
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/dq7ev0ez",
    },
}

TaskAwareHalfCheetahDir_v0 = {
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/q72fbi9c",
    },
}

AntGoal_v0 = {
    "train": {
        "batch_size": 64,
    },
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/r61mx5i0",
    },
}

TaskAwareAntGoal_v0 = {
    "train": {
        "batch_size": 64,
    },
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/31w4e9ws",
    },
}
