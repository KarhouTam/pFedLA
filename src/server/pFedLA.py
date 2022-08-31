from path import Path

_CURRENT_DIR = Path(__file__).parent.abspath()
import sys

sys.path.append(_CURRENT_DIR.parent)
sys.path.append(_CURRENT_DIR.parent / "data")


import os
import pickle
import random
from collections import OrderedDict
from typing import List, OrderedDict, Tuple

import torch
from client.pFedLA import pFedLAClient
from rich.progress import track
from tqdm import tqdm
from utils.models import HyperNetwork
from utils.args import get_pFedLA_args

from base import ServerBase


class pFedLAServer(ServerBase):
    def __init__(self):
        super(pFedLAServer, self).__init__(get_pFedLA_args(), "pFedLA")
        self.log_name = "{}_{}_{}_{}_{}.html".format(
            self.algo,
            self.args.dataset,
            self.args.global_epochs,
            self.args.local_epochs,
            self.args.k,
        )

        self.logger.log("Initializing clients model...")
        passed_epoch = 0
        if os.listdir(self.temp_dir) != []:
            self.client_model_params_list = torch.load(
                self.temp_dir / "clients_model.pt"
            )
            with open(self.temp_dir / "epoch.pkl", "rb") as f:
                passed_epoch = pickle.load(f)
            self.logger.log(
                "Find existed clients model...",
                f"Have run {passed_epoch} epochs already.",
            )
        else:
            self.client_model_params_list = [
                list(self.backbone(self.args.dataset).state_dict().values())
                for _ in range(self.client_num_in_total)
            ]
        self.global_epochs = self.args.global_epochs - passed_epoch
        _dummy_model = self.backbone(self.args.dataset)
        self.logger.log("Backbone:", _dummy_model)
        self.hypernet = HyperNetwork(
            embedding_dim=self.args.embedding_dim,
            client_num=self.client_num_in_total,
            hidden_dim=self.args.hidden_dim,
            backbone=_dummy_model,
            K=self.args.k,
            gpu=self.args.gpu,
        )

        self.trainer = pFedLAClient(
            backbone=_dummy_model,
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            valset_ratio=self.args.valset_ratio,
            testset_ratio=self.args.testset_ratio,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )

        self.all_params_name = [name for name in _dummy_model.state_dict().keys()]
        self.trainable_params_name = [
            name
            for name, param in _dummy_model.state_dict(keep_vars=True).items()
            if param.requires_grad
        ]
        self.all_clients_stats = {i: {} for i in self.client_id_indices}

    def train(self) -> None:
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
            for client_id in selected_clients:
                (
                    client_local_params,
                    retain_blocks,
                ) = self.generate_client_model_parameters(client_id)
                diff, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    verbose=(E % self.args.verbose_gap) == 0,
                )
                self.all_clients_stats[client_id][f"ROUND: {E}"] = (
                    f"retain {retain_blocks}, {stats['loss_before']:.4f} -> {stats['loss_after']:.4f}",
                )

                self.update_hypernetwork(client_id, diff, retain_blocks)

                self.update_client_model_parameters(client_id, diff)

            if E % self.args.save_period == 0:
                torch.save(
                    self.client_model_params_list, self.temp_dir / "clients_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
        self.logger.log(self.all_clients_stats)

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
            client_local_params, retain_blocks = self.generate_client_model_parameters(
                client_id
            )
            dummy_diff, stats = self.trainer.test(
                client_id=client_id, model_params=client_local_params,
            )

            # NOTE: make sure that all client model params are on CPU, not CUDA
            # or self.generate_...() would raise the error of stacking tensors on different devices
            self.update_client_model_parameters(client_id, dummy_diff)
            self.logger.log(
                f"client [{client_id}] retain {retain_blocks}, [red]loss: {stats['loss']:.4f}    [magenta]accuracy: {stats['acc']:.2f}%"
            )
            all_loss.append(stats["loss"])
            all_acc.append(stats["acc"])

        self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
        self.logger.log(
            "loss: {:.4f}    accuracy: {:.2f}%".format(
                sum(all_loss) / len(all_loss), sum(all_acc) / len(all_acc),
            )
        )

    @torch.no_grad()
    def update_client_model_parameters(
        self, client_id: int, delta: OrderedDict[str, torch.Tensor],
    ) -> None:
        updated_params = []
        for param, diff in zip(
            self.client_model_params_list[client_id], delta.values()
        ):
            updated_params.append((param + diff).detach().cpu())
        self.client_model_params_list[client_id] = updated_params

    def generate_client_model_parameters(
        self, client_id: int
    ) -> Tuple[OrderedDict[str, torch.Tensor], List[str]]:
        layer_params_dict = dict(
            zip(self.all_params_name, list(zip(*self.client_model_params_list)))
        )
        alpha, retain_blocks = self.hypernet(client_id)
        aggregated_parameters = {}
        default_weight = torch.tensor(
            [i == client_id for i in range(self.client_num_in_total)],
            dtype=torch.float,
            device=self.device,
        )
        for name in self.all_params_name:
            if name in self.trainable_params_name:
                a = alpha[name.split(".")[0]]
            else:
                a = default_weight
            if a.sum() == 0:
                self.logger.log(self.all_clients_stats)
                raise RuntimeError(
                    f"client [{client_id}]'s {name.split('.')[0]} alpha is a all 0 vector"
                )
            aggregated_parameters[name] = torch.sum(
                a
                / a.sum()
                * torch.stack(layer_params_dict[name], dim=-1).to(self.device),
                dim=-1,
            )

        self.client_model_params_list[client_id] = list(aggregated_parameters.values())
        return aggregated_parameters, retain_blocks

    def update_hypernetwork(
        self,
        client_id: int,
        diff: OrderedDict[str, torch.Tensor],
        retain_blocks: List[str] = [],
    ) -> None:
        # calculate gradients
        hn_grads = torch.autograd.grad(
            outputs=list(
                filter(
                    lambda param: param.requires_grad,
                    self.client_model_params_list[client_id],
                )
            ),
            inputs=self.hypernet.mlp_parameters()
            + self.hypernet.fc_layer_parameters()
            + self.hypernet.emd_parameters(),
            grad_outputs=list(
                map(
                    lambda tup: tup[1],
                    filter(
                        lambda tup: tup[1].requires_grad
                        and tup[0].split(".")[0] not in retain_blocks,
                        diff.items(),
                    ),
                )
            ),
            allow_unused=True,
        )
        mlp_grads = hn_grads[: len(self.hypernet.mlp_parameters())]
        fc_grads = hn_grads[
            len(self.hypernet.mlp_parameters()) : len(
                self.hypernet.mlp_parameters() + self.hypernet.fc_layer_parameters()
            )
        ]
        emd_grads = hn_grads[
            len(self.hypernet.mlp_parameters() + self.hypernet.fc_layer_parameters()) :
        ]

        for param, grad in zip(self.hypernet.fc_layer_parameters(), fc_grads):
            if grad is not None:
                param.data -= self.args.hn_lr * grad

        for param, grad in zip(self.hypernet.mlp_parameters(), mlp_grads):
            param.data -= self.args.hn_lr * grad

        for param, grad in zip(self.hypernet.emd_parameters(), emd_grads):
            param.data -= self.args.hn_lr * grad

        self.hypernet.save_hn()

    def run(self):
        super().run()
        # clean out all HNs
        self.hypernet.clean_models()


if __name__ == "__main__":
    server = pFedLAServer()
    server.run()
