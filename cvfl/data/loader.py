import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Dict

def get_dataset(dataset_name: str = "mnist"):
    """
    Load raw vision dataset (MNIST, Fashion-MNIST, CIFAR-10).
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
    elif dataset_name in ["fashion-mnist", "fashion_mnist", "fmnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_set = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        in_channels, num_classes = 3, 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_set, test_set, in_channels, num_classes


def partition_data_iid(dataset: Dataset, num_clients: int) -> List[Subset]:
    """
    IID Random uniform partitioning across N clients.
    """
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    subsets = []
    for i in range(num_clients):
        chosen = list(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(chosen))
        subsets.append(Subset(dataset, chosen))
    return subsets


def partition_data_dirichlet(dataset: Dataset, num_clients: int, alpha: float = 0.5) -> List[Subset]:
    """
    Non-IID partitioning using Dirichlet distribution Dir(alpha).
    Lower alpha -> higher label heterogeneity (more non-IID).
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([y for _, y in dataset])

    num_classes = len(np.unique(labels))
    client_id_map = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_c_i) < len(dataset) / num_clients) for p, idx_c_i in zip(proportions, client_id_map)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)

        for i in range(num_clients):
            client_id_map[i].extend(splits[i])

    for i in range(num_clients):
        np.random.shuffle(client_id_map[i])

    subsets = [Subset(dataset, idxs) for idxs in client_id_map]
    return subsets
