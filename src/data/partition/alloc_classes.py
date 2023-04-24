import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def randomly_alloc_classes(
    ori_datasets: List[Dataset],
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    transform=None,
    target_transform=None,
) -> Tuple[List[List[int]], Dict[str, Dict[str, int]]]:
    data_indices = [[] for _ in range(num_clients)]
    data_numpy = np.concatenate(
        [ds.data for ds in ori_datasets], axis=0, dtype=np.int64
    )
    targets_numpy = np.concatenate(
        [ds.targets for ds in ori_datasets], axis=0, dtype=np.int64
    )
    classes_label = list(range(len(np.unique(targets_numpy))))
    idx = [np.where(targets_numpy == i)[0].tolist() for i in classes_label]
    assigned_classes = [[] for _ in range(num_clients)]
    selected_classes = classes_label
    if num_classes * num_clients > len(selected_classes):
        selected_classes.extend(
            np.random.choice(
                classes_label, num_classes * num_clients - len(selected_classes)
            ).tolist()
        )
    random.shuffle(selected_classes)
    for i, cls in enumerate(range(0, num_clients * num_classes, num_classes)):
        assigned_classes[i] = selected_classes[cls : cls + num_classes]

    selected_times = Counter(selected_classes[: num_clients * num_classes])
    labels_count = Counter(targets_numpy)
    batch_size = np.zeros_like(classes_label)

    for cls in selected_times.keys():
        batch_size[cls] = int(labels_count[cls] / selected_times[cls])

    for i in range(num_clients):
        for cls in assigned_classes[i]:
            selected_idx = random.sample(idx[cls], batch_size[cls])
            data_indices[i] = np.concatenate(
                [data_indices[i], selected_idx], axis=0
            ).astype(np.int64)
            idx[cls] = list(set(idx[cls]) - set(selected_idx))

        data_indices[i] = data_indices[i].tolist()

    stats = {}
    datasets = []
    for i, indices in enumerate(data_indices):
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
