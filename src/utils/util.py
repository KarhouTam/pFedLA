import json
import math
import os
import pickle
import random
from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple, Union

import numpy as np
import torch
from path import Path
from torch.utils.data import DataLoader, Subset, random_split

PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
LOG_DIR = PROJECT_DIR / "logs"
TEMP_DIR = PROJECT_DIR / "temp"
DATASETS_DIR = PROJECT_DIR / "datasets"


def get_dataloader(
    dataset: str,
    client_id: int,
    batch_size=20,
    valset_ratio=0.2,
    testset_ratio=0.1,
    only_dataset=False,
) -> Dict[str, Union[DataLoader, Subset]]:
    args_dict = json.load(open(DATASETS_DIR / "args.json", "r"))
    client_num_in_each_pickles = args_dict["client_num_in_each_pickles"]
    pickles_dir = DATASETS_DIR / dataset / "pickles"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    pickle_path = (
        pickles_dir / f"{math.floor(client_id / client_num_in_each_pickles)}.pkl"
    )
    with open(pickle_path, "rb") as f:
        subset = pickle.load(f)
    client_dataset = subset[client_id % client_num_in_each_pickles]
    val_samples_num = int(len(client_dataset) * valset_ratio)
    test_samples_num = int(len(client_dataset) * testset_ratio)
    train_samples_num = len(client_dataset) - val_samples_num - test_samples_num
    trainset, valset, testset = random_split(
        client_dataset, [train_samples_num, val_samples_num, test_samples_num]
    )
    if only_dataset:
        return {"train": trainset, "val": valset, "test": testset}
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size)
    testloader = DataLoader(testset, batch_size)
    return {"train": trainloader, "val": valloader, "test": testloader}


def get_client_id_indices(
    dataset,
) -> Union[Tuple[List[int], List[int], int], Tuple[List[int], int]]:
    args_dict = json.load(open(DATASETS_DIR / "args.json", "r"))
    dataset_pickles_path = DATASETS_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    if args_dict["type"] == "user":
        return seperation["train"], seperation["test"], seperation["total"]
    else:  # NOTE: "sample"
        return seperation["id"], seperation["total"]


def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


