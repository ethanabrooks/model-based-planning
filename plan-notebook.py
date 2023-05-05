# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

import re
import time

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from environments.navigation.point_robot import semi_circle_goal_sampler
from trajectory import utils
from trajectory.search import make_prefix
from trajectory.search.sampling import sample_n
from trajectory.utils import Parser as UtilsParser
from trajectory.utils import load_model
from trajectory.utils.setup import set_seed
from trajectory.utils.timer import Timer
from utils.writer import Writer

from trajectory.datasets import (  # isort: skip
    get_preprocess_fn,
    load_environment,
    local,
)


# %%
beam_width = 128
cdf_act = 0.6
cdf_obs = None
commit = "68fbb3c19d80e400175990d6cce551b1abfa0c26 main"
config = "config.offline"
dataset = "SparsePointEnv-v0"
debug = True
device = "cuda"
exp_name = "plans/defaults/freq1_H15_beam128"
gpt_epoch = "latest"
gpt_loadpath = "gpt/azure"
horizon = 15
k_act = None
k_obs = 1
loadpath = "rldl/In-Context Model-Based Planning/bcetko3o"
logbase = "logs/"
max_context_transitions = 10
n_expand = 2
name = "plan"
notes = ""
percentile = "mean"
plan_freq = 1
prefix = "plans/defaults/"
prefix_context = True
renderer = "PointRenderer"
seed = 42
suffix = "0"
total_iters = 200
trajectory_transformer = False
verbose = True
vis_freq = 5
args = dict(locals())
args

# %%


class Parser(UtilsParser):
    dataset: str = "halfcheetah-medium-expert-v2"
    config: str = "config.offline"
    debug: bool = False
    name: str = "plan"
    notes: str = None
    trajectory_transformer: bool = False


# assert "SparsePointEnv-v0" in dataset
# %% [markdown]
# ####### setup ########
# ######################
# %%
assert any(name in dataset for name in ["SparsePointEnv", "HalfCheetahVel"])

set_seed(seed)

writer = Writer.make(
    debug, config=args, dataset=dataset, name=name, notes=notes, run=None
)

# %% [markdown]
# ###### models ########
# ######################

# %%
sleep_time = 1
while True:
    try:
        wandb.restore("discretizer.pkl", run_path=loadpath, root=writer.directory)
        break
    except wandb.errors.CommError as e:
        print(e)
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
        sleep_time *= 2

env = dataset
discretizer = writer.load_pickle("discretizer.pkl")

wandb.restore("model_config.pkl", run_path=loadpath, root=writer.directory)
api = wandb.Api()
for run_file in api.run(loadpath).files():
    if re.match("state_[0-9]+.pt", run_file.name):
        wandb.restore(run_file.name, run_path=loadpath, root=writer.directory)
gpt, gpt_epoch = load_model(
    writer.directory,
    epoch=gpt_epoch,
    device=device,
)

# %% [markdown]
# ###### env #######
# ######################

# %%
task_aware = local.is_task_aware(env)
env = local.get_env_name(env)
env = load_environment(env)
if task_aware:
    env = local.TaskWrapper(env)
# renderer = make_renderer(**args, env=env)
timer = Timer()

discount = discretizer.discount
observation_dim = discretizer.observation_dim
action_dim = discretizer.action_dim

value_fn = lambda x: discretizer.value_fn(x, percentile)
preprocess_fn = get_preprocess_fn(env.spec.id)

# %% [markdown]
# ##### main loop ######
# ######################

# %%
observation = env.reset()
total_reward = 0

## observations for rendering
rollout = [observation.copy()]

## previous (tokenized) transitions for conditioning transformer
context = []

terminal_mdp = True  # trigger visualization on first timestep
t = T = 0


# %%
prefix = make_prefix(discretizer, context, observation, prefix_context)

# %%
max_context_transitions = None
k_obs = None
k_act = None
k_rew = 1
cdf_obs = None
cdf_act = None
cdf_rew = None
verbose = True
previous_actions = None
model = gpt
value_fn = value_fn
x = prefix
n_steps = horizon
beam_width = beam_width
n_expand = n_expand
observation_dim = observation_dim
action_dim = action_dim
discount = discount
max_context_transitions = max_context_transitions
verbose = verbose
k_obs = k_obs
k_act = k_act
cdf_obs = cdf_obs
cdf_act = cdf_act

# %%

REWARD_DIM = VALUE_DIM = 1
# convert max number of transitions to max number of tokens
transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
max_block = (
    max_context_transitions * transition_dim - 1 if max_context_transitions else None
)

## pass in max numer of tokens to sample function
sample_kwargs = {
    "max_block": max_block,
    "crop_increment": transition_dim,
}

## repeat input for search
x = x.repeat(beam_width, 1)

## construct reward and discount tensors for estimating values
rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)
discounts = discount ** torch.arange(n_steps + 1, device=x.device)

## logging
progress = utils.Progress(n_steps) if verbose else utils.Silent()

# %%
print(torch.cuda.is_available())
torch.cuda.device_count()

# %%
t = 0
## repeat everything by `n_expand` before we sample actions
x = x.repeat(n_expand, 1)
rewards = rewards.repeat(n_expand, 1)

## sample actions
x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

## sample reward and value estimate
x, r_probs = sample_n(
    model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs
)

## optionally, use a percentile or mean of the reward and
## value distributions instead of sampled tokens
r_t, V_t = value_fn(r_probs)

## update rewards tensor
rewards[:, t] = r_t
rewards[:, t + 1] = V_t

## estimate values using rewards up to `t` and terminal value at `t`
values = (rewards * discounts).sum(dim=-1)

## get `beam_width` best actions
values, inds = torch.topk(values, beam_width)

## index into search candidates to retain `beam_width` highest-reward sequences
x = x[inds]
rewards = rewards[inds]

# %%
sequence_recon = discretizer.reconstruct(x.reshape(-1, transition_dim))
sequence_recon = sequence_recon.reshape(beam_width, t + 1, transition_dim)
sequence_recon.shape

# %%
## sample next observation (unless we have reached the end of the planning horizon)
if t < n_steps - 1:
    x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)
t += 1
## repeat everything by `n_expand` before we sample actions
x = x.repeat(n_expand, 1)
rewards = rewards.repeat(n_expand, 1)

## sample actions
x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

## sample reward and value estimate
x, r_probs = sample_n(
    model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs
)

## optionally, use a percentile or mean of the reward and
## value distributions instead of sampled tokens
r_t, V_t = value_fn(r_probs)

## update rewards tensor
rewards[:, t] = r_t
rewards[:, t + 1] = V_t

## estimate values using rewards up to `t` and terminal value at `t`
values = (rewards * discounts).sum(dim=-1)

## get `beam_width` best actions
values, inds = torch.topk(values, beam_width)

## index into search candidates to retain `beam_width` highest-reward sequences
x = x[inds]
rewards = rewards[inds]

# %%
r_probs[0, 0]

# %%

plt.plot(r_probs[1, 0].cpu().numpy())

# %%

figsize = (5.5, 4)
fig, axis = plt.subplots(1, 1, figsize=figsize)
xlim = (-1.3, 1.3)

## Necessary because of insane circular import error

if env.unwrapped.goal_sampler == semi_circle_goal_sampler:
    ylim = (-0.3, 1.3)
else:
    ylim = (-1.3, 1.3)
curr_task = env.get_task()

# plot goal
axis.scatter(*curr_task, marker="x", color="k", s=50)

# radius where we get reward
if hasattr(env, "goal_radius"):
    circle1 = plt.Circle(
        curr_task, env.goal_radius, color="c", alpha=0.2, edgecolor="none"
    )
    plt.gca().add_artist(circle1)

# plot (semi-)circle
r = 1.0
if env.unwrapped.goal_sampler == semi_circle_goal_sampler:
    angle = np.linspace(0, np.pi, 100)
else:
    angle = np.linspace(0, 2 * np.pi, 100)
goal_range = r * np.array((np.cos(angle), np.sin(angle)))
plt.plot(goal_range[0], goal_range[1], "k--", alpha=0.1)

# Normalize the data to the range 0 to 1
norm = colors.Normalize(vmin=values.min(), vmax=values.max())

# Choose a colormap
cmap = cm.get_cmap("Blues")

# Create a ScalarMappable object that maps normalized data to colors
value_sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# Normalize the data to the range 0 to 1
norm = colors.Normalize(vmin=rewards.min(), vmax=rewards.max())

# Choose a colormap
cmap = cm.get_cmap("Greens")
reward_sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# plot trajectory
for states, value, reward in zip(
    sequence_recon[:, :, :2], values.cpu().numpy(), rewards[:, t].cpu().numpy()
):
    axis.plot(states[:, 0], states[:, 1], "-", color=value_sm.to_rgba(value))
    if reward > 0:
        axis.scatter(*states[-1], color=reward_sm.to_rgba(reward), marker="x")

plt.xlim(xlim)
plt.ylim(ylim)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
