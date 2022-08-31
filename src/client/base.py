from collections import OrderedDict
from copy import deepcopy

import torch
from path import Path
from rich.console import Console
from torch.utils.data import DataLoader

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)

from utils.util import get_dataloader


class ClientBase:
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        valset_ratio: float,
        testset_ratio: float,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.client_id: int = None
        self.valset: DataLoader = None
        self.trainset: DataLoader = None
        self.testset: DataLoader = None
        self.optimizer: torch.optim.Optimizer = None
        self.model: torch.nn.Module = deepcopy(backbone)
        self.dataset = dataset
        self.batch_size = batch_size
        self.valset_ratio = valset_ratio
        self.testset_ratio = testset_ratio
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        size = 0
        loss = 0
        correct = 0
        for x, y in self.testset:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)

            loss += self.criterion(logits, y)

            pred = torch.softmax(logits, -1).argmax(-1)

            correct += (pred == y).int().sum()

            size += y.size(-1)

        acc = correct / size * 100.0
        loss = loss / len(self.testset)
        return loss, acc

    def train(self):
        pass

    def get_client_local_dataset(self):
        datasets = get_dataloader(
            self.dataset,
            self.client_id,
            self.batch_size,
            self.valset_ratio,
            self.testset_ratio,
        )
        self.trainset = datasets["train"]
        self.valset = datasets["val"]
        self.testset = datasets["test"]

    def _log_while_training(self, evaluate=True, verbose=False):
        def _log_and_train(*args, **kwargs):
            loss_before = 0
            loss_after = 0
            acc_before = 0
            acc_after = 0
            if evaluate:
                loss_before, acc_before = self.evaluate()

            res = self._train(*args, **kwargs)

            if evaluate:
                loss_after, acc_after = self.evaluate()

            if verbose:
                self.logger.log(
                    "client [{}]   [bold red]loss: {:.4f} -> {:.4f}    [bold blue]accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id, loss_before, loss_after, acc_before, acc_after
                    )
                )

            eval_stats = {
                "loss_before": loss_before,
                "loss_after": loss_after,
                "acc_before": acc_before,
                "acc_after": acc_after,
            }
            return res, eval_stats

        return _log_and_train

    def _train(self):
        pass

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
