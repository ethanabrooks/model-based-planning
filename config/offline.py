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
        "action_weight": 5,
        "attn_pdrop": 0.1,
        "batch_size": 256,
        "device": "cuda",
        "discount": 0.99,
        "discretizer": "QuantileDiscretizer",
        "embd_pdrop": 0.1,
        "exp_name": gpt_expname,
        "learning_rate": 6e-4,
        "logbase": logbase,
        "lr_decay": True,
        "N": 100,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 4,
        "n_saves": 3,
        "resid_pdrop": 0.1,
        "reward_weight": 1,
        "seed": 42,
        "step": 1,
        "subsampled_sequence_length": 50,
        "termination_penalty": -100,
        "total_iters": 200_000,
        "value_weight": 1,
    },
    "plan": {
        "beam_width": 128,
        "cdf_act": 0.6,
        "cdf_obs": None,
        "device": "cuda",
        "exp_name": watch(args_to_watch),
        "gpt_epoch": "latest",
        "gpt_loadpath": gpt_expname,
        "horizon": 15,
        "k_act": None,
        "k_obs": 1,
        "logbase": logbase,
        "max_context_transitions": 10,
        "percentile": "mean",
        "plan_freq": 1,
        "renderer": "Renderer",
        "prefix": "plans/defaults/",
        "prefix_context": True,
        "seed": 42,
        "suffix": "0",
        "total_episodes": 10,
        "verbose": True,
        "vis_freq": 50,
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

sparse_point_env_train = {
    "batch_size": 200,
    "subsampled_sequence_length": 100,
}

SparsePointEnv_v0 = {
    "train": sparse_point_env_train,
    "plan": {
        "renderer": "PointRenderer",
        "loadpath": "rldl/In-Context Model-Based Planning/bcetko3o",
    },
}

TaskAwareSparsePointEnv_v0 = {
    "train": sparse_point_env_train,
    "plan": {
        "renderer": "PointRenderer",
        "loadpath": "rldl/In-Context Model-Based Planning/edioi44c",
    },
}


mujoco_train = {
    "batch_size": 64,
}

mujoco_plan = {
    "total_episodes": 3,
}

HalfCheetahVel_v0 = {
    "train": mujoco_train,
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/an22r3ig",
        **mujoco_plan,
    },
}

TaskAwareHalfCheetahVel_v0 = {
    "train": mujoco_train,
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/u2g0tsra",
        **mujoco_plan,
    },
}

HalfCheetahDir_v0 = {
    "train": mujoco_train,
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/hdvq9xsm",
        **mujoco_plan,
    },
}

TaskAwareHalfCheetahDir_v0 = {
    "train": mujoco_train,
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/q72fbi9c",
        **mujoco_plan,
    },
}

AntDir2D_v0 = {
    "train": {"batch_size": 32},
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/1tpgl4z5",
        **mujoco_plan,
    },
}

TaskAwareAntDir2D_v0 = {
    "train": {"batch_size": 32},
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/31w4e9ws",
        **mujoco_plan,
    },
}

AntGoal_v0 = {
    "train": {"batch_size": 32},
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/1tpgl4z5",
        **mujoco_plan,
    },
}

TaskAwareAntGoal_v0 = {
    "train": {"batch_size": 32},
    "plan": {
        "loadpath": "rldl/In-Context Model-Based Planning/31w4e9ws",
        **mujoco_plan,
    },
}
