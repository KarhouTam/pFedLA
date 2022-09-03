from argparse import ArgumentParser, Namespace


def get_pFedLA_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--global_epochs", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--hn_lr", type=float, default=5e-3)
    parser.add_argument("--verbose_gap", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "emnist", "fmnist"],
        default="cifar10",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--valset_ratio", type=float, default=0.0)
    parser.add_argument("--testset_ratio", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--save_period", type=int, default=20)
    return parser.parse_args()


def get_FedAvg_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--local_lr", type=float, default=1e-2)
    parser.add_argument("--verbose_gap", type=int, default=20)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "emnist", "fmnist"],
        default="cifar10",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--valset_ratio", type=float, default=0.0)
    parser.add_argument("--testset_ratio", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--save_period", type=int, default=20)
    return parser.parse_args()
