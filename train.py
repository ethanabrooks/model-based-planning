import itertools
import math

import torch
import torch.nn as nn
import wandb
from rich import box
from rich.table import Table
from wandb.sdk.wandb_run import Run

import trajectory.utils as utils
from trajectory.models.transformers import GPT
from trajectory.utils.git_utils import get_relative_git_rev
from trajectory.utils.setup import console
from utils import helpers
from utils.writer import Writer

import trajectory.datasets as datasets  # isort: skip


class Parser(utils.Parser):
    dataset: str = "halfcheetah-medium-expert-v2"
    config: str = "config.offline"
    results_log_dir: str = None
    debug: bool = False
    name: str = "train"
    notes: str = None
    target_commit: str = None
    trajectory_transformer: bool = False
    action_mask: bool = False


def main(
    action_mask: bool,
    args: dict,
    action_weight: float,
    attn_pdrop: float,
    batch_size: int,
    dataset: str,
    debug: bool,
    device: int,
    discount: float,
    discretizer: str,
    embd_pdrop: float,
    exp_name: str,
    learning_rate: float,
    lr_decay: float,
    N: int,
    name: str,
    notes: str,
    n_embd: int,
    n_layer: int,
    n_head: int,
    save_freq: int,
    resid_pdrop: float,
    run: Run,
    reward_weight: float,
    step: int,
    subsampled_sequence_length: int,
    target_commit: str,
    termination_penalty: float,
    trajectory_transformer: bool,
    total_iters: int,
    value_weight: float,
    **_,
):
    if target_commit is not None:
        args.update(relative_commit=get_relative_git_rev(target_commit=target_commit))

    writer = Writer.make(
        debug,
        config=args,
        dataset=dataset,
        name=name,
        notes=notes,
        run=run,
        trajectory_transformer=trajectory_transformer,
        action_mask=action_mask,
    )

    #######################
    ####### dataset #######
    #######################

    sequence_length = subsampled_sequence_length * step

    dataset_config = utils.Config(
        datasets.DiscretizedDataset,
        savepath=str(writer.path("data_config.pkl")),
        env=dataset,
        N=N,
        penalty=termination_penalty,
        sequence_length=sequence_length,
        step=step,
        discount=discount,
        discretizer=discretizer,
        trajectory_transformer=trajectory_transformer,
        action_mask=action_mask,
    )
    dataset = dataset_config()
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = dataset.joined_dim

    discretizer_fname = "discretizer.pkl"
    writer.dump_pickle(dataset.discretizer, discretizer_fname)
    writer.save(writer.path(discretizer_fname))

    #######################
    ######## model ########
    #######################

    block_size = subsampled_sequence_length * transition_dim - 1
    table = Table(show_header=False, box=box.HORIZONTALS)
    for k, v in {
        "Dataset size": f"{len(dataset)}",
        "Joined dim": f"{transition_dim}",
        "Block size": f"{block_size}",
        "Observation dim": f"{obs_dim}",
        "Action dim": f"{act_dim}",
        "block_size": f"{block_size}",
    }.items():
        table.add_row(k, v)
    console.print(table)

    model_config = utils.Config(
        GPT,
        savepath=str(writer.path("model_config.pkl")),
        ## discretization
        vocab_size=N,
        block_size=block_size,
        ## architecture
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd * n_head,
        ## dimensions
        observation_dim=obs_dim,
        action_dim=act_dim,
        transition_dim=transition_dim,
        ## loss weighting
        action_weight=action_weight,
        reward_weight=reward_weight,
        value_weight=value_weight,
        ## dropout probabilities
        embd_pdrop=embd_pdrop,
        resid_pdrop=resid_pdrop,
        attn_pdrop=attn_pdrop,
    )
    writer.save(model_config.savepath)

    model = model_config()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    #######################
    ####### trainer #######
    #######################

    final_tokens = (
        total_iters * block_size * batch_size
    )  # number of tokens seen during training
    warmup_tokens = final_tokens // 20

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=str(writer.path("trainer_config.pkl")),
        # optimization parameters
        batch_size=batch_size,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        grad_norm_clip=1.0,
        weight_decay=0.1,  # only applied on matmul weights
        # learning rate decay: linear warmup followed by cosine decay to 10% of original
        lr_decay=lr_decay,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        ## dataloader
        num_workers=0,
        device=device,
        total_iters=total_iters,
    )
    writer.save(trainer_config.savepath)
    trainer = trainer_config()

    #######################
    ###### main loop ######
    #######################

    ## scale number of epochs to keep number of updates constant
    console.log(f"\n{dataset} | {exp_name}")

    for epoch in itertools.count():
        cuml_it = trainer.train(
            model=model,
            dataset=dataset,
            debug=debug,
            save_freq=save_freq,
            writer=writer,
        )
        if cuml_it >= total_iters:
            break

    wandb.finish()


def get_args():
    return Parser().parse_args(experiment="train")


def sweep(**config):
    args = get_args()
    return helpers.sweep(
        main,
        parser=args,
        param_space=config,
        group_name=args.name,
        dataset=args.dataset,
        gpus_per_proc=1,
    )


if __name__ == "__main__":
    ARGS = get_args().as_dict()
    main(**ARGS, args=ARGS, run=None)
