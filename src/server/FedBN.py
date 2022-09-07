import os
import pickle
import random
from typing import Dict

import torch
from rich.progress import track
from tqdm import tqdm

from base import ServerBase
from client.FedBN import FedBNClient
from utils.args import get_FedAvg_args
from utils.util import clone_parameters


class FedBNServer(ServerBase):
    def __init__(self):
        super(FedBNServer, self).__init__(get_FedAvg_args(), "FedBN")
        self.log_name = "{}_{}_{}_{}.html".format(
            self.algo,
            self.args.dataset,
            self.args.global_epochs,
            self.args.local_epochs,
        )

        self.trainer = FedBNClient(
            backbone=self.backbone(self.args.dataset),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            valset_ratio=self.args.valset_ratio,
            testset_ratio=self.args.testset_ratio,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )
        self.clients_bn_stats: Dict[int, Dict[str, torch.Tensor]] = {}

    def train(self):
        if os.path.exists(self.temp_dir / "client_bn_stats.pkl"):
            with open(self.temp_dir / "client_bn_stats.pkl", "rb") as f:
                self.clients_bn_stats = pickle.load(f)

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
                client_bn_stats = (
                    self.clients_bn_stats[client_id]
                    if client_id in self.clients_bn_stats.keys()
                    else None
                )
                (updated_params, weight), stats = self.trainer.train(
                    client_id=client_id,
                    bn_stats=client_bn_stats,
                    model_params=client_local_params,
                    verbose=(E % self.args.verbose_gap) == 0,
                )

                # save clients bn stats
                bn_stats = {}
                for name, param in zip(self.global_params_dict.keys(), updated_params):
                    if "bn" in name:
                        bn_stats[name] = param.clone()
                self.clients_bn_stats[client_id] = bn_stats

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
                with open(self.temp_dir / "clients_bn_stats.pkl", "wb") as f:
                    pickle.dump(self.clients_bn_stats, f)
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
        self.logger.log(self.all_clients_stats)


if __name__ == "__main__":
    server = FedBNServer()
    server.run()
