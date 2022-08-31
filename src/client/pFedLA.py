from collections import OrderedDict
from typing import OrderedDict

import torch
from rich.console import Console
from utils.util import clone_parameters

from .base import ClientBase


class pFedLAClient(ClientBase):
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
        super(pFedLAClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            valset_ratio,
            testset_ratio,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        verbose=True,
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.model = self.model.to(self.device)
        res, stats = self._log_while_training(evaluate=True, verbose=verbose)()
        self.model = self.model.cpu()
        return res, stats

    def _train(self):
        self.model.train()
        frz_model_params = clone_parameters(self.model)
        dataset = self.trainset
        for x, y in dataset:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)

            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        delta = OrderedDict(
            {
                k: p1 - p0
                for (k, p1), p0 in zip(
                    self.model.state_dict(keep_vars=True).items(),
                    frz_model_params.values(),
                )
            }
        )
        return delta

    def test(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.model = self.model.to(self.device)
        loss, acc = self.evaluate()
        dummy_diff = OrderedDict(
            {
                name: torch.zeros_like(param)
                for name, param in self.model.state_dict().items()
            }
        )
        self.model.cpu()
        stats = {"loss": loss, "acc": acc}
        return dummy_diff, stats
