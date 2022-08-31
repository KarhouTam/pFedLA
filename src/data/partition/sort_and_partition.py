import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def sort_and_alloc(
    datasets: List[Dataset], num_clients: int, num_classes: int
) -> Dict[int, np.ndarray]:
    total_sample_nums = sum(map(lambda ds: len(ds), datasets))
    num_shards = num_clients * num_classes
    # one shard's length indicate how many data samples that belongs to one class that one client can obtain.
    size_of_shards = int(total_sample_nums / num_shards)

    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_clients)}

    labels = np.concatenate([ds.targets for ds in datasets], axis=0, dtype=np.int64)
    idxs = np.arange(total_sample_nums)

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = random.sample(idx_shard, num_classes)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (
                    dict_users[i],
                    idxs[rand * size_of_shards : (rand + 1) * size_of_shards],
                ),
                axis=0,
            )

    return dict_users


def randomly_alloc_classes(
    ori_datasets: List[Dataset],
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    transform=None,
    target_transform=None,
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    stats = {}
    dict_users = sort_and_alloc(ori_datasets, num_clients, num_classes)
    targets_numpy = np.concatenate(
        [ds.targets for ds in ori_datasets], axis=0, dtype=np.int64
    )
    data_numpy = np.concatenate(
        [ds.data for ds in ori_datasets], axis=0, dtype=np.float32
    )
    datasets = []
    for i, indices in dict_users.items():
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(indices)
        stats[i]["y"] = Counter(targets_numpy[indices].tolist())
        datasets.append(
            target_dataset(
                data=data_numpy[indices],
                targets=targets_numpy[indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    return datasets, stats
