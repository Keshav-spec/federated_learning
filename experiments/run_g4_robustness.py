import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
from typing import Dict, Any

from cvfl.models.cnn import SimpleCNN
from cvfl.data.loader import get_dataset, partition_data_dirichlet
from cvfl.federated.client import CVFLClient
from cvfl.federated.edge import EdgeAggregator
from cvfl.federated.server import CloudServer
from cvfl.attacks.poisoning import PoisoningAttacker


def evaluate_robustness_under_poisoning(
    dataset_name: str = "fashion-mnist",
    num_clients: int = 10,
    rounds: int = 10,
    samples_per_client: int = 1000,
    alpha: float = 0.5,
    local_epochs: int = 2
) -> Dict[str, Any]:
    """
    Evaluates G4 Robustness: Global Accuracy vs Malicious Client Ratio (0%-40%)
    under Label-Flip, Sign-Flip, and Gaussian Noise Byzantine Attacks.

    Utilizes CV-FL Adaptive SignScore Byzantine-Robust Filtering with
    Dual-Layer Secret Masking and Merkle Spot-Check Verification.
    """
    print(f"\n---> Loading {dataset_name.upper()} Dataset for G4 Robustness Benchmark...")
    print(f"  - Config: {num_clients} Clients | {rounds} Rounds | {samples_per_client} Samples/Client")
    print(f"  - Local Epochs: {local_epochs} | Adaptive SignScore-Weighted FedAvg Aggregation")
    sys.stdout.flush()

    train_set, test_set, in_channels, num_classes = get_dataset(dataset_name)

    # Fast in-memory tensor caching for ultra-fast, robust evaluation
    if dataset_name.lower() in ["fashion-mnist", "fashion_mnist", "fmnist"]:
        test_x = (test_set.data.float().unsqueeze(1) / 255.0 - 0.2860) / 0.3530
        test_y = test_set.targets.long()
        train_x_all = (train_set.data.float().unsqueeze(1) / 255.0 - 0.2860) / 0.3530
        train_y_all = train_set.targets.long()
    elif dataset_name.lower() == "mnist":
        test_x = (test_set.data.float().unsqueeze(1) / 255.0 - 0.1307) / 0.3081
        test_y = test_set.targets.long()
        train_x_all = (train_set.data.float().unsqueeze(1) / 255.0 - 0.1307) / 0.3081
        train_y_all = train_set.targets.long()
    else:
        test_x = torch.stack([x for x, _ in test_set][:2000])
        test_y = torch.tensor([y for _, y in test_set][:2000])
        train_x_all = torch.stack([x for x, _ in train_set])
        train_y_all = torch.tensor([y for _, y in train_set])

    eval_size = min(2000, len(test_y))
    test_loader = DataLoader(TensorDataset(test_x[:eval_size], test_y[:eval_size]), batch_size=512, shuffle=False)

    client_subsets = partition_data_dirichlet(train_set, num_clients=num_clients, alpha=alpha)

    # Pre-cache client subsets in memory with class-balanced Dirichlet shuffling
    client_tensor_data = []
    for i in range(num_clients):
        idxs = np.array(client_subsets[i].indices)
        np.random.seed(42 + i)
        np.random.shuffle(idxs)
        sub_idxs = idxs[:min(samples_per_client, len(idxs))]
        client_tensor_data.append((train_x_all[sub_idxs], train_y_all[sub_idxs]))

    malicious_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
    attack_types = ["label_flip", "sign_flip", "gaussian_noise"]
    results = {"malicious_ratios": malicious_ratios, "experiments": {}}

    for attack in attack_types:
        print(f"\n[G4 EVALUATION] Attack Type: '{attack.upper()}'")
        sys.stdout.flush()
        results["experiments"][attack] = {}

        for mal_ratio in malicious_ratios:
            num_malicious = int(num_clients * mal_ratio)
            print(f"\n  * Malicious Ratio: {int(mal_ratio*100)}% ({num_malicious}/{num_clients} Malicious Nodes)")
            sys.stdout.flush()

            start_t = time.time()

            clients = []
            for i in range(num_clients):
                is_malicious = (i < num_malicious)
                attacker = PoisoningAttacker(attack_type=attack, attack_scale=2.0) if is_malicious else None
                c_x, c_y = client_tensor_data[i]
                c_dataset = TensorDataset(c_x, c_y)
                client = CVFLClient(
                    client_id=i,
                    dataset=c_dataset,
                    batch_size=128,
                    lr=0.003,
                    lr_decay=0.95,
                    optimizer_type="adam",
                    attacker=attacker
                )
                clients.append(client)

            model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
            ea = EdgeAggregator(temperature=12.0)
            server = CloudServer(model=model, spot_check_ratio=0.1)

            final_acc = 0.0
            for r in range(1, rounds + 1):
                client_outputs = [
                    c.train_epoch(server.model, epochs=local_epochs, round_num=r - 1)
                    for c in clients
                ]

                m_list = [o[0] for o in client_outputs]
                c_list = [o[1] for o in client_outputs]
                tr_list = [o[2] for o in client_outputs]
                s_list = [o[3] for o in client_outputs]
                r1_list = [o[4] for o in client_outputs]
                r2_list = [o[5] for o in client_outputs]
                w_list = [o[6] for o in client_outputs]

                edge_agg, scores, weights, commitments, merkle_trees = ea.aggregate_edge(
                    client_masked_scalars=m_list,
                    client_r2_masks=r2_list,
                    client_unmasked_signs=s_list,
                    client_commitments=c_list,
                    merkle_trees=tr_list,
                    adaptive_filter=True
                )

                server.process_round(
                    edge_masked_agg=edge_agg,
                    client_r1_masks=r1_list,
                    weights=weights,
                    client_commitments=commitments,
                    merkle_trees=merkle_trees,
                    original_d=len(s_list[0]),
                    client_signs=s_list,
                    client_local_weights=w_list
                )

                _, acc = server.evaluate(test_loader)
                final_acc = acc
                print(f"    Round {r:2d}/{rounds}: Test Acc={acc*100:.2f}% | Malicious Weight={np.sum(weights[:num_malicious])*100:.2f}%", flush=True)

            elapsed = time.time() - start_t
            print(f"  --> DONE in {elapsed:.1f}s | Final Accuracy: {final_acc*100:.2f}%\n", flush=True)
            results["experiments"][attack][f"mal_{int(mal_ratio*100)}pc"] = round(final_acc * 100.0, 2)

    return results


if __name__ == "__main__":
    res = evaluate_robustness_under_poisoning()
    print(json.dumps(res, indent=2))
