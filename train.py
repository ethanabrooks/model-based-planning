import math
import os

import torch
import wandb
from rich import box
from rich.table import Table
from wandb.sdk.wandb_run import Run

import trajectory.utils as utils
from trajectory.models.transformers import GPT
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
    trajectory_transformer: bool = False


def main(
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
    n_saves: int,
    resid_pdrop: float,
    run: Run,
    reward_weight: float,
    step: int,
    subsampled_sequence_length: int,
    termination_penalty: float,
    trajectory_transformer: bool,
    total_iters: int,
    value_weight: float,
    **_,
):
    assert any(name in dataset for name in ["SparsePointEnv", "HalfCheetahVel"])
    writer = Writer.make(
        debug, config=args, dataset=dataset, name=name, notes=notes, run=run
    )

    #######################
    ####### dataset #######
    #######################

    sequence_length = subsampled_sequence_length * step

    dataset_config = utils.Config(
        datasets.DiscretizedDataset,
        savepath=writer.path("data_config.pkl"),
        env=dataset,
        N=N,
        penalty=termination_penalty,
        sequence_length=sequence_length,
        step=step,
        discount=discount,
        discretizer=discretizer,
        trajectory_transformer=trajectory_transformer,
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
        savepath=writer.path("model_config.pkl"),
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
        savepath=writer.path("trainer_config.pkl"),
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
    n_epochs = math.ceil(total_iters / len(dataset))
    save_freq = math.ceil(n_epochs / n_saves)

    for epoch in range(n_epochs):
        print(f"\nEpoch: {epoch} / {n_epochs} | {dataset} | {exp_name}")

        trainer.train(model, dataset, debug)

        ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
        save_epoch = (epoch + 1) // save_freq * save_freq
        statepath = os.path.join(writer.directory, f"state_{save_epoch}.pt")
        print(f"Saving model to {statepath}")

        ## save state to disk
        state = model.state_dict()
        torch.save(state, statepath)
        writer.save(statepath)

    wandb.finish()


def get_args():
    return Parser().parse_args("train")


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
    ARGS = get_args().as_dict()
    main(**ARGS, args=ARGS, run=None)
