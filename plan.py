import json
import re
import time
from os.path import join

import wandb
from rich.console import Console
from wandb.sdk.wandb_run import Run

from trajectory.search import beam_plan, extract_actions, make_prefix, update_context
from trajectory.utils import Parser as UtilsParser
from trajectory.utils import load_model, make_renderer
from trajectory.utils.setup import set_seed
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
    trajectory_transformer: bool = False
    baseline: str = None
    n_expand: int = 2


def main(
    args: dict,
    baseline: str,
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
    seed: int,
    suffix: str,
    total_episodes: int,
    trajectory_transformer: bool,
    verbose: bool,
    vis_freq: int,
    **_,
):
    #######################
    ######## setup ########
    #######################

    set_seed(seed)

    writer = Writer.make(
        debug,
        config=args,
        dataset=dataset,
        name=name,
        notes=notes,
        run=run,
        trajectory_transformer=trajectory_transformer,
        baseline=baseline,
    )
    console = Console()

    if baseline == "ad":
        beam_width = 1
        n_expand = 1
    elif baseline == "ad++":
        beam_width *= n_expand
        n_expand = 1
    elif baseline is not None:
        raise ValueError(f"Unknown baseline: {baseline}")

    #######################
    ####### models ########
    #######################

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

    #######################
    ####### env #######
    #######################

    task_aware = local.is_task_aware(env)
    env = local.get_env_name(env)
    env = load_environment(env)
    try:
        env.seed(seed)
    except AttributeError:
        pass

    while True:
        task = env.sample_task()
        is_test = env.test_task_mask(task[None]).item()
        if is_test:
            env.set_task(task)
            break

    if task_aware:
        env = local.TaskWrapper(env)
    renderer = make_renderer(**args, env=env)

    discount = discretizer.discount
    observation_dim = discretizer.observation_dim
    action_dim = discretizer.action_dim

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
    e = t = T = 0

    while True:
        observation = preprocess_fn(observation)

        if t % plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, prefix_context)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                model=gpt,
                value_fn=value_fn,
                x=prefix,
                n_steps=horizon,
                beam_width=beam_width,
                n_expand=n_expand,
                observation_dim=observation_dim,
                action_dim=action_dim,
                discount=discount,
                max_context_transitions=max_context_transitions,
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
                join(writer.save_directory, f"{t}_plan.mp4"),
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
        terminal_mdp = bool(info.get("done_mdp"))
        log = {
            "reward": reward,
            "total_reward": total_reward,
            "score": score,
        }
        if terminal_mdp:
            log["episode_return"] = total_reward
        writer.log(
            log,
            step=T,
        )
        console.log(
            dict(
                **log,
                **{
                    "episode step": t,
                    "total step": T,
                    "episode": e,
                    "elapsed steps": env.env.env.env.env.env._elapsed_steps,
                },
            )
        )

        ## visualization
        if terminal or terminal_mdp:
            ## save rollout thus far
            renderer.render_rollout(
                join(writer.save_directory, "rollout.mp4"), rollout, fps=80
            )

        t += 1
        T += 1
        if terminal_mdp:
            e += 1
            if trajectory_transformer:
                context = []
            rollout = []
            t = 0
            total_reward = 0
            if e == total_episodes:
                break
        if terminal:
            break

        observation = next_observation

    ## save result as a json file
    json_path = join(writer.save_directory, "rollout.json")
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
