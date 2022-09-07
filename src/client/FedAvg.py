from collections import OrderedDict
from typing import OrderedDict

import torch
from rich.console import Console

from .base import ClientBase


class FedAvgClient(ClientBase):
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
        super(FedAvgClient, self).__init__(
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
        self.model.to(self.device)
        res, stats = self._log_while_training(evaluate=True, verbose=verbose)()
        self.model.cpu()
        return res, stats

    def _train(self):
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.trainset:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return list(self.model.state_dict().values()), len(self.trainset.dataset)

    def test(
        self, client_id: int, model_params: OrderedDict[str, torch.Tensor],
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        self.model.to(self.device)
        loss, acc = self.evaluate()
        self.model.cpu()
        stats = {"loss": loss, "acc": acc}
        return stats
