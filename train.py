import json
import os
import numpy as np
import torch
import wandb

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.models.transformers import GPT
from utils.helpers import project_name


class Parser(utils.Parser):
    dataset: str = "halfcheetah-medium-expert-v2"
    config: str = "config.offline"
    results_log_dir: str = None
    debug: bool = False
    name: str = None


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
    n_embd: int,
    n_epochs_ref: int,
    n_layer: int,
    n_head: int,
    n_saves: int,
    resid_pdrop: float,
    reward_weight: float,
    step: int,
    subsampled_sequence_length: int,
    termination_penalty: float,
    value_weight: float,
    **_,
):
    #######################
    ######## setup ########
    #######################

    if not debug:
        name = f"train-{dataset}" if name is None else name
        wandb.init(
            project=project_name(),
            name=name,
            config=args,
        )
        with open(os.path.join(wandb.run.dir, "config.json"), "w") as f:
            config = {
                k: v
                for k, v in args.as_dict().items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            config.update(device=device.type)
            json.dump(config, f, indent=2)

    #######################
    ####### dataset #######
    #######################

    sequence_length = subsampled_sequence_length * step

    dataset_config = utils.Config(
        datasets.DiscretizedDataset,
        savepath=None if wandb.run is None else (wandb.run.dir, "data_config.pkl"),
        env=dataset,
        N=N,
        penalty=termination_penalty,
        sequence_length=sequence_length,
        step=step,
        discount=discount,
        discretizer=discretizer,
    )
    if dataset_config.savepath:
        wandb.save(dataset_config.savepath)

    dataset = dataset_config()
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = dataset.joined_dim

    #######################
    ######## model ########
    #######################

    block_size = subsampled_sequence_length * transition_dim - 1
    print(
        f"Dataset size: {len(dataset)} | "
        f"Joined dim: {transition_dim} "
        f"(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}"
    )

    model_config = utils.Config(
        GPT,
        savepath=None if wandb.run is None else (wandb.run.dir, "model_config.pkl"),
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
    if model_config.savepath:
        wandb.save(model_config.savepath)

    model = model_config()
    model.to(device)

    #######################
    ####### trainer #######
    #######################

    warmup_tokens = len(dataset) * block_size  ## number of tokens seen per epoch
    final_tokens = 20 * warmup_tokens

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=None if wandb.run is None else (wandb.run.dir, "trainer_config.pkl"),
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
    )
    if trainer_config.savepath:
        wandb.save(trainer_config.savepath)

    trainer = trainer_config()

    #######################
    ###### main loop ######
    #######################

    ## scale number of epochs to keep number of updates constant
    n_epochs = int(1e6 / len(dataset) * n_epochs_ref)
    save_freq = int(n_epochs // n_saves)

    for epoch in range(n_epochs):
        print(f"\nEpoch: {epoch} / {n_epochs} | {dataset} | {exp_name}")

        trainer.train(model, dataset, debug)

        ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
        save_epoch = (epoch + 1) // save_freq * save_freq
        if wandb.run is not None:
            statepath = os.path.join(wandb.run.dir, f"state_{save_epoch}.pt")
            print(f"Saving model to {statepath}")

            ## save state to disk
            state = model.state_dict()
            torch.save(state, statepath)
            wandb.save(statepath)

    wandb.finish()


if __name__ == "__main__":
    args = Parser().parse_args("train")
    args = args.as_dict()
    main(**args, args=args)
