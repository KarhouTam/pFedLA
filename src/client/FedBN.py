from typing import Dict, OrderedDict

import torch
from rich.console import Console

from .FedAvg import FedAvgClient


class FedBNClient(FedAvgClient):
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

        self.bn_stats: Dict[str, torch.Tensor] = {}

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        bn_stats: Dict[str, torch.Tensor],
        verbose=True,
    ):
        self.bn_stats = bn_stats
        return super().train(client_id, model_params, verbose)

    def set_parameters(self, model_params: OrderedDict[str, torch.Tensor]):
        self.model.load_state_dict(model_params, strict=True)
        if self.bn_stats is not None:
            self.model.load_state_dict(self.bn_stats, strict=False)
