import json
import re
import time
from os.path import join

from wandb.sdk.wandb_run import Run

import wandb
from trajectory.search import beam_plan, extract_actions, make_prefix, update_context
from trajectory.utils import Parser as UtilsParser
from trajectory.utils import load_from_config, load_model, make_renderer
from trajectory.utils.timer import Timer
from utils import helpers
from utils.writer import Writer

from trajectory.datasets import (  # isort: skip
    get_preprocess_fn,
    load_environment,
    local,
)


class Parser(UtilsParser):
    dataset: str = "halfcheetah-medium-expert-v2"
    config: str = "config.offline"
    debug: bool = False
    name: str = "plan"
    notes: str = None


def main(
    args: dict,
    beam_width: int,
    cdf_obs: float,
    cdf_act: float,
    dataset: str,
    debug: bool,
    device: int,
    exp_name: str,
    gpt_epoch: int,
    horizon: int,
    k_act: int,
    k_obs: int,
    loadpath: str,
    max_context_transitions: int,
    name: str,
    n_expand: int,
    notes: str,
    percentile: float,
    plan_freq: int,
    prefix_context: int,
    run: Run,
    suffix: str,
    verbose: bool,
    vis_freq: int,
    **_,
):
    assert dataset == "TaskAwareSparsePointEnv-v0"
    #######################
    ######## setup ########
    #######################

    writer = Writer.make(
        debug, config=args, dataset=dataset, name=name, notes=notes, run=run
    )

    #######################
    ####### models ########
    #######################

    sleep_time = 1
    while True:
        try:
            wandb.restore("data_config.pkl", run_path=loadpath, root=writer.directory)
            break
        except wandb.errors.CommError as e:
            print(e)
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            sleep_time *= 2

    env = dataset
    dataset = load_from_config(writer.directory, "data_config.pkl")

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

    #######################
    ####### dataset #######
    #######################

    task_aware = local.is_task_aware(env)
    env = local.get_env_name(env)
    env = load_environment(env)
    if task_aware:
        env = local.TaskWrapper(env)
    renderer = make_renderer(**args, env=env)
    timer = Timer()

    discretizer = dataset.discretizer
    discount = dataset.discount
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    value_fn = lambda x: discretizer.value_fn(x, percentile)
    preprocess_fn = get_preprocess_fn(env.spec.id)

    #######################
    ###### main loop ######
    #######################

    observation = env.reset()
    total_reward = 0

    ## observations for rendering
    rollout = [observation.copy()]

    ## previous (tokenized) transitions for conditioning transformer
    context = []

    terminal_mdp = True  # trigger visualization on first timestep
    t = T = 0

    while True:
        observation = preprocess_fn(observation)

        if t % plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, prefix_context)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                gpt,
                value_fn,
                prefix,
                horizon,
                beam_width,
                n_expand,
                observation_dim,
                action_dim,
                discount,
                max_context_transitions,
                verbose=verbose,
                k_obs=k_obs,
                k_act=k_act,
                cdf_obs=cdf_obs,
                cdf_act=cdf_act,
            )

        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)

        ## visualization
        if t % vis_freq == 0 or terminal_mdp:
            ## save current plan
            renderer.render_plan(
                join(writer.directory, f"{t}_plan.mp4"),
                sequence_recon,
                env.state_vector(),
            )

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

        ## execute action in environment
        next_observation, reward, terminal, info = env.step(action)

        ## update return
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ## update rollout observations and context transitions
        rollout.append(next_observation.copy())
        context = update_context(
            context,
            discretizer,
            observation,
            action,
            reward,
            max_context_transitions,
        )

        writer.log(
            {
                "reward": reward,
                "total_reward": total_reward,
                "score": score,
            },
            step=T,
        )
        print(
            f"[ plan ] t: {T} / {env.spec.max_episode_steps} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | "
            f"time: {timer():.2f} | {dataset} | {exp_name} | {suffix}\n"
        )

        terminal_mdp = bool(info.get("done_mdp"))

        ## visualization
        if t % vis_freq == 0 or terminal or terminal_mdp:
            ## save rollout thus far
            renderer.render_rollout(
                join(writer.directory, "rollout.mp4"), rollout, fps=80
            )

        t += 1
        T += 1
        if terminal_mdp:
            context = []
            rollout = []
            t = 0
        if terminal:
            break

        observation = next_observation

    ## save result as a json file
    json_path = join(writer.directory, "rollout.json")
    json_data = {
        "score": score,
        "step": t,
        "return": total_reward,
        "term": terminal,
        "gpt_epoch": gpt_epoch,
    }
    json.dump(json_data, open(json_path, "w"), indent=2, sort_keys=True)


def get_args():
    return Parser().parse_args("plan")


def sweep(**config):
    args = get_args()
    return helpers.sweep(
        main,
        parser=args,
        param_space=config,
        group_name=args.name,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    args = get_args()
    ARGS = get_args().as_dict()
    main(**ARGS, args=ARGS, run=None)
