from path import Path

_CURRENT_DIR = Path(__file__).parent.abspath()
import sys

sys.path.append(_CURRENT_DIR.parent)

import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, FashionMNIST

from dataset import CIFARDataset, MNISTDataset
from partition import dirichlet_distribution, randomly_alloc_classes
from utils.util import DATASETS_DIR

DATASET = {
    "fmnist": (FashionMNIST, MNISTDataset),
    "cifar10": (CIFAR10, CIFARDataset),
    "cifar100": (CIFAR100, CIFARDataset),
    "emnist": (EMNIST, MNISTDataset),
}

MEAN = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4865, 0.4409),
    "emnist": (0.1736,),
    "fmnist": (0.2860,),
}

STD = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2009, 0.1984, 0.2023),
    "emnist": (0.3248,),
    "fmnist": (0.3205,),
}


def main(args):
    dataset_root = (
        Path(args.root).abspath() / args.dataset
        if args.root is not None
        else DATASETS_DIR / args.dataset
    )
    pickles_dir = DATASETS_DIR / args.dataset / "pickles"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose(
        [transforms.Normalize(MEAN[args.dataset], STD[args.dataset])]
    )
    target_transform = None

    os.makedirs(dataset_root, exist_ok=True)
    if os.path.isdir(pickles_dir):
        os.system(f"rm -rf {pickles_dir}")
    os.mkdir(pickles_dir)

    client_num_in_total = args.client_num_in_total
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "emnist":
        trainset = ori_dataset(
            dataset_root,
            train=True,
            download=True,
            split=args.emnist_split,
            transform=transforms.ToTensor(),
        )
        testset = ori_dataset(
            dataset_root,
            train=False,
            split=args.emnist_split,
            transform=transforms.ToTensor(),
        )
    else:
        trainset = ori_dataset(dataset_root, train=True, download=True)
        testset = ori_dataset(dataset_root, train=False)
    concat_datasets = [trainset, testset]
    if args.alpha > 0:  # NOTE: Dirichlet(alpha)
        all_datasets, stats = dirichlet_distribution(
            ori_dataset=concat_datasets,
            target_dataset=target_dataset,
            num_clients=client_num_in_total,
            alpha=args.alpha,
            transform=transform,
            target_transform=target_transform,
        )
    else:  # NOTE: sort and partition
        classes = len(ori_dataset.classes) if args.classes <= 0 else args.classes
        all_datasets, stats = randomly_alloc_classes(
            ori_datasets=concat_datasets,
            target_dataset=target_dataset,
            num_clients=client_num_in_total,
            num_classes=max(1, classes),
            transform=transform,
            target_transform=target_transform,
        )

    for subset_id, client_id in enumerate(
        range(0, len(all_datasets), args.client_num_in_each_pickles)
    ):
        subset = all_datasets[client_id : client_id + args.client_num_in_each_pickles]
        with open(pickles_dir / str(subset_id) + ".pkl", "wb") as f:
            pickle.dump(subset, f)

    # save stats
    client_id_indices = [i for i in range(client_num_in_total)]
    with open(pickles_dir / "seperation.pkl", "wb") as f:
        pickle.dump({"id": client_id_indices, "total": client_num_in_total}, f)
    with open(DATASETS_DIR / args.dataset / "all_stats.json", "w") as f:
        json.dump(stats, f)

    args.root = (
        Path(args.root).abspath()
        if str(dataset_root) != str(DATASETS_DIR / args.dataset)
        else None
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fmnist", "cifar10", "cifar100", "emnist"],
        default="cifar10",
    )
    ################# Dirichlet distribution only #################
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Only for controling data hetero degree while performing Dirichlet partition.",
    )
    ###############################################################
    parser.add_argument("--client_num_in_total", type=int, default=200)
    parser.add_argument(
        "--classes",
        type=int,
        default=-1,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument("--seed", type=int, default=0)

    ################# For EMNIST only #####################
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )
    #######################################################
    parser.add_argument("--client_num_in_each_pickles", type=int, default=10)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()
    main(args)
    args_dict = dict(args._get_kwargs())
    with open(DATASETS_DIR / "args.json", "w") as f:
        json.dump(args_dict, f)
