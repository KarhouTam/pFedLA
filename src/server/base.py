from collections import OrderedDict
import os
from argparse import Namespace
import random

import torch
from path import Path
from rich.console import Console
from rich.progress import track
import pickle
from tqdm import tqdm

_CURRENT_DIR = Path(__file__).parent.abspath()

import sys

sys.path.append(_CURRENT_DIR.parent)


from utils.models import CNNWithBatchNorm, CNNWithoutBatchNorm
from utils.util import (
    LOG_DIR,
    TEMP_DIR,
    clone_parameters,
    fix_random_seed,
    get_client_id_indices,
)
from client.base import ClientBase


class ServerBase:
    def __init__(self, args: Namespace, algo: str):
        self.algo = algo
        self.args = args
        # default log file format
        self.log_name = "{}_{}_{}_{}.html".format(
            self.algo,
            self.args.dataset,
            self.args.global_epochs,
            self.args.local_epochs,
        )
        self.device = torch.device(
            "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
        )
        fix_random_seed(self.args.seed)
        self.backbone = (
            CNNWithBatchNorm
            if self.args.dataset in ["cifar10", "cifar100"]
            else CNNWithoutBatchNorm
        )
        self.logger = Console(record=True, log_path=False, log_time=False,)
        self.client_id_indices, self.client_num_in_total = get_client_id_indices(
            self.args.dataset
        )
        self.temp_dir = TEMP_DIR / self.algo
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        _dummy_model = self.backbone(self.args.dataset)
        passed_epoch = 0
        self.global_params_dict: OrderedDict[str : torch.Tensor] = None
        if os.listdir(self.temp_dir) != []:
            if os.path.exists(self.temp_dir / "global_model.pt"):
                self.global_params_dict = torch.load(self.temp_dir / "global_model.pt")
                self.logger.log("Find existed global model...")

            if os.path.exists(self.temp_dir / "epoch.pkl"):
                with open(self.temp_dir / "epoch.pkl", "rb") as f:
                    passed_epoch = pickle.load(f)
                self.logger.log(f"Have run {passed_epoch} epochs already.",)
        else:
            self.global_params_dict = OrderedDict(_dummy_model.state_dict())

        self.global_epochs = self.args.global_epochs - passed_epoch
        self.logger.log("Backbone:", _dummy_model)

        self.trainer: ClientBase = None
        self.all_clients_stats = {i: {} for i in self.client_id_indices}

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )
        for E in progress_bar:

            if E % self.args.verbose_gap == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            updated_params_cache = []
            weights_cache = []
            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)
                (updated_params, weight), stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    verbose=(E % self.args.verbose_gap) == 0,
                )

                updated_params_cache.append(updated_params)
                weights_cache.append(weight)
                self.all_clients_stats[client_id][f"ROUND: {E}"] = (
                    f"{stats['loss_before']:.4f} -> {stats['loss_after']:.4f}",
                )

            self.aggregate_parameters(updated_params_cache, weights_cache)

            if E % self.args.save_period == 0:
                torch.save(
                    self.global_params_dict, self.temp_dir / "global_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
        self.logger.log(self.all_clients_stats)

    @torch.no_grad()
    def aggregate_parameters(self, updated_params_cache, weights_cache):
        weight_sum = sum(weights_cache)
        weights = torch.tensor(weights_cache, device=self.device) / weight_sum

        aggregated_params = []

        for params in zip(*updated_params_cache):
            aggregated_params.append(
                torch.sum(weights * torch.stack(params, dim=-1), dim=-1)
            )

        self.global_params_dict = OrderedDict(
            zip(self.global_params_dict.keys(), aggregated_params)
        )

    def test(self) -> None:
        self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
        all_loss = []
        all_acc = []
        for client_id in track(
            self.client_id_indices,
            "[bold blue]Testing...",
            console=self.logger,
            disable=self.args.log,
        ):
            client_local_params = clone_parameters(self.global_params_dict)
            stats = self.trainer.test(
                client_id=client_id, model_params=client_local_params,
            )

            self.logger.log(
                f"client [{client_id}]  [red]loss: {stats['loss']:.4f}    [magenta]accuracy: {stats['acc']:.2f}%"
            )
            all_loss.append(stats["loss"])
            all_acc.append(stats["acc"])

        self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
        self.logger.log(
            "loss: {:.4f}    accuracy: {:.2f}%".format(
                sum(all_loss) / len(all_loss), sum(all_acc) / len(all_acc),
            )
        )

    def run(self):
        self.logger.log("Arguments:", dict(self.args._get_kwargs()))
        self.train()
        self.test()
        if self.args.log:
            if not os.path.isdir(LOG_DIR):
                os.mkdir(LOG_DIR)
            self.logger.save_html(LOG_DIR / self.log_name)

        # delete all temporary files
        if os.listdir(self.temp_dir) != []:
            os.system(f"rm -rf {self.temp_dir}")
