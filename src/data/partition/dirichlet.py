from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def dirichlet_distribution(
    ori_dataset: List[Dataset],
    target_dataset: Dataset,
    num_clients: int,
    alpha: float,
    transform=None,
    target_transform=None,
) -> Tuple[List[Dataset], Dict]:
    NUM_CLASS = len(ori_dataset[0].classes)
    MIN_SIZE = 0
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    stats = {}
    targets_numpy = np.concatenate(
        [ds.targets for ds in ori_dataset], axis=0, dtype=np.int64
    )
    data_numpy = np.concatenate(
        [ds.data for ds in ori_dataset], axis=0, dtype=np.float32
    )
    idx = [np.where(targets_numpy == i)[0] for i in range(NUM_CLASS)]

    while MIN_SIZE < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]
            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            stats[i] = {"x": None, "y": None}
            np.random.shuffle(idx_batch[i])
            X[i] = data_numpy[idx_batch[i]]
            Y[i] = targets_numpy[idx_batch[i]]
            stats[i]["x"] = len(X[i])
            stats[i]["y"] = Counter(Y[i].tolist())

    datasets = [
        target_dataset(
            data=X[j],
            targets=Y[j],
            transform=transform,
            target_transform=target_transform,
        )
        for j in range(num_clients)
    ]
    return datasets, stats
