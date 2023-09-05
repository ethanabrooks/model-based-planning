from typing import Optional

import numpy as np
import torch
from gym import Env

from trajectory.search.sampling import get_logp, sample_n, sort_2d
from trajectory.utils import rendering
from trajectory.utils.discretization import QuantileDiscretizer

REWARD_DIM = VALUE_DIM = 1


def get_transition_dim(observation_dim, action_dim):
    return observation_dim + action_dim + REWARD_DIM + VALUE_DIM


def get_max_block(max_context_transitions, transition_dim):
    return (
        max_context_transitions * transition_dim - 1
        if max_context_transitions
        else None
    )


@torch.no_grad()
def beam_plan(
    model,
    value_fn,
    x: torch.Tensor,
    n_steps: int,
    beam_width: int,
    ground_truth_reward: bool,
    ground_truth_state: bool,
    n_expand: int,
    observation_dim: int,
    action_dim: int,
    discretizer: QuantileDiscretizer,
    env: Env,
    discount: float = 0.99,
    max_context_transitions: Optional[int] = None,
    k_obs: Optional[int] = None,
    k_act: Optional[int] = None,
    k_rew: int = 1,
    cdf_obs: Optional[float] = None,
    cdf_act: Optional[float] = None,
    cdf_rew: Optional[float] = None,
    verbose: bool = True,
):
    """
    x : tensor[ 1 x input_sequence_length ]
    """

    # convert max number of transitions to max number of tokens
    transition_dim = get_transition_dim(observation_dim, action_dim)
    max_block = get_max_block(max_context_transitions, transition_dim)

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

    ## ground truth
    try:
        step_fn = env.step_fn
    except AttributeError:
        step_fn = None
    use_ground_truth = step_fn is not None and (
        ground_truth_state or ground_truth_reward
    )
    x2 = None

    for t in range(n_steps):
        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        [_, r_ind] = x.shape

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

        if use_ground_truth:
            x_t = x.view(n_expand * beam_width, -1, transition_dim)[:, -1]
            sequence_recon = discretizer.reconstruct(x_t)
            o, a, _, _ = rendering.split(sequence_recon, observation_dim, action_dim)
            o2, r2, _, _ = step_fn(o, a)
            r2 = r2[..., None]
            V2 = np.zeros_like(r2)  # dummy for discretization
            joined = np.concatenate([o2, a, r2, V2], axis=-1)
            joined_discrete = discretizer.discretize(joined)
            x2, _, r2, _ = rendering.split(joined_discrete, observation_dim, action_dim)
            if ground_truth_reward:
                r2 = torch.tensor(r2).to(x.device)
                rewards[:, t] = r2
                x[:, r_ind] = r2

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        ## index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            [_, x_ind] = x.shape
            x, _ = sample_n(
                model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs
            )
            if x2 is not None and ground_truth_state:
                x2 = torch.tensor(x2).to(x.device)
                x[:, x_ind:] = x2[inds]

    ## [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]

    ## return best sequence
    argmax = values.argmax()
    best_sequence = x[argmax]

    return best_sequence


@torch.no_grad()
def beam_search(model, x, n_steps, beam_width=512, goal=None, **sample_kwargs):
    batch_size = len(x)

    prefix_i = torch.arange(len(x), dtype=torch.long, device=x.device)
    cumulative_logp = torch.zeros(batch_size, 1, device=x.device)

    for t in range(n_steps):
        if goal is not None:
            goal_rep = goal.repeat(len(x), 1)
            logp = get_logp(model, x, goal=goal_rep, **sample_kwargs)
        else:
            logp = get_logp(model, x, **sample_kwargs)

        candidate_logp = cumulative_logp + logp
        sorted_logp, sorted_i, sorted_j = sort_2d(candidate_logp)

        n_candidates = (candidate_logp > -np.inf).sum().item()
        n_retain = min(n_candidates, beam_width)
        cumulative_logp = sorted_logp[:n_retain].unsqueeze(-1)

        sorted_i = sorted_i[:n_retain]
        sorted_j = sorted_j[:n_retain].unsqueeze(-1)

        x = torch.cat([x[sorted_i], sorted_j], dim=-1)
        prefix_i = prefix_i[sorted_i]

    x = x[0]
    return x, cumulative_logp.squeeze()
