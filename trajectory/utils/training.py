import math
import os
from matplotlib import pyplot as plt

import torch
import wandb
from rich.console import Console
from torch.utils.data.dataloader import DataLoader

from utils.helpers import print_row
import torch.nn.functional as F


def to(xs, device):
    return [x.to(device) for x in xs]


console = Console()


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.n_epochs = 0
        self.n_tokens = 0  # counter used for learning rate decay
        self.optimizer = None

    def get_optimizer(self, model):
        if self.optimizer is None:
            console.log(f"Making optimizer at epoch {self.n_epochs}")
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def train(self, model, dataset, debug, save_freq, writer, n_epochs=1, log_freq=100):
        config = self.config
        optimizer = self.get_optimizer(model)
        model.train(True)
        vocab_size = dataset.N

        loader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        for _ in range(n_epochs):
            for it, batch in enumerate(loader):
                X, _, _ = batch
                X = F.pad(X, (0,1), "constant", 0)
                discretizer = dataset.discretizer
                X = X.reshape(config.batch_size, -1, discretizer.observation_dim + discretizer.action_dim + 2)
                x, y = X[:, :, :2].permute(2, 0, 1)
                x = x.numpy()
                y = y.numpy()
                for i in range(32):
                    plt.plot(x[i], y[i], color='blue', alpha=0.1) # Set alpha value as per your requirement
                if it % log_freq == 0:
                    plt.savefig("lines.png")
                    breakpoint()
                continue
                # batch = to(batch, self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    logits, loss = model(*batch)

                # backprop and update the parameters
                model.zero_grad()
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_norm_clip
                )
                optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    y = batch[-2]
                    self.n_tokens += (
                        y != vocab_size
                    ).sum()  # number of tokens processed this step
                    if self.n_tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(
                            max(1, config.warmup_tokens)
                        )
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - config.warmup_tokens) / float(
                            max(1, config.final_tokens - config.warmup_tokens)
                        )
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    lr = config.learning_rate

                if it % save_freq == 0:
                    ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
                    statepath = os.path.join(writer.directory, f"state_{it}.pt")
                    print(f"Saving model to {statepath}")

                    ## save state to disk
                    state = model.state_dict()
                    torch.save(state, statepath)
                    writer.save(statepath)

                # report progress
                if it % log_freq == 0:
                    cuml_it = it + len(loader) * self.n_epochs
                    _, targets, mask = batch
                    argmax_accuracy = logits.argmax(-1) == targets
                    argmax_accuracy = argmax_accuracy[mask].float().mean()
                    probs = torch.softmax(logits[0], dim=-1)
                    [exp_accuracy] = torch.gather(
                        probs, dim=-1, index=targets[0, :, None]
                    ).T  # just use first batch index for speed
                    exp_accuracy = exp_accuracy[mask[0]].float().mean()
                    norms = [p.norm().item() for p in model.parameters()]
                    global_norm = sum(x**2 for x in norms) ** 0.5
                    log = {
                        "train loss": loss.item(),
                        "argmax accuracy": argmax_accuracy.item(),
                        "exp accuracy": exp_accuracy.item(),
                        "global norm": global_norm,
                        "lr": lr,
                        "lr_mult": lr_mult,
                        "epoch": self.n_epochs,
                        "iteration": it,
                    }
                    print_row(
                        log,
                        show_header=it % (log_freq * 30) == 0,
                        format={
                            "train loss": lambda x: f"{x:.2f}",
                            "argmax accuracy": lambda x: f"{x:.2%}",
                            "exp accuracy": lambda x: f"{x:.2%}",
                            "global norm": lambda x: f"{x:.2f}",
                            "lr": lambda x: f"{x:.2e}",
                            "lr_mult": lambda x: f"{x:.2f}",
                            "iteration": lambda x: f"{x:,}/{config.total_iters:,} ({x / config.total_iters:.1%})",
                        },
                        widths=dict(iteration=0.2),
                    )
                    if not debug:
                        wandb.log(log, step=cuml_it)

                if cuml_it >= config.total_iters:
                    break

            self.n_epochs += 1
