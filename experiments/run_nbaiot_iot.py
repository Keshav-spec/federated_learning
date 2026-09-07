import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import json
from typing import Dict, Any

from cvfl.models.mlp import TabularMLP
from cvfl.data.nbaiot import generate_simulated_nbaiot_devices
from cvfl.federated.client import CVFLClient
from cvfl.federated.edge import EdgeAggregator
from cvfl.federated.server import CloudServer


def evaluate_nbaiot_iot_intrusion(
    num_devices: int = 9,
    rounds: int = 15,
    local_epochs: int = 2
) -> Dict[str, Any]:
    """
    Evaluates CV-FL on simulated N-BaIoT IoT intrusion telemetry across device nodes.
    """
    device_datasets, global_test_dataset = generate_simulated_nbaiot_devices(
        num_devices=num_devices, samples_per_device=800, feature_dim=115
    )

    test_loader = torch.utils.data.DataLoader(global_test_dataset, batch_size=128, shuffle=False)

    clients = [
        CVFLClient(client_id=i, dataset=device_datasets[i], batch_size=64, lr=0.01)
        for i in range(num_devices)
    ]

    model = TabularMLP(input_dim=115, hidden_dim=128, num_classes=2)
    ea = EdgeAggregator()
    server = CloudServer(model=model, spot_check_ratio=0.1)

    round_history = []

    for r in range(1, rounds + 1):
        client_outputs = [
            c.train_epoch(server.model, epochs=local_epochs, round_num=r - 1)
            for c in clients
        ]

        m_list = [o[0] for o in client_outputs]
        c_list = [o[1] for o in client_outputs]
        tree_list = [o[2] for o in client_outputs]
        sign_list = [o[3] for o in client_outputs]
        r1_list = [o[4] for o in client_outputs]
        r2_list = [o[5] for o in client_outputs]
        w_list = [o[6] for o in client_outputs]

        edge_agg, scores, weights, commitments, merkle_trees = ea.aggregate_edge(
            client_masked_scalars=m_list,
            client_r2_masks=r2_list,
            client_unmasked_signs=sign_list,
            client_commitments=c_list,
            merkle_trees=tree_list
        )

        res = server.process_round(
            edge_masked_agg=edge_agg,
            client_r1_masks=r1_list,
            weights=weights,
            client_commitments=commitments,
            merkle_trees=merkle_trees,
            original_d=len(sign_list[0]),
            client_signs=sign_list,
            client_local_weights=w_list
        )

        loss, acc = server.evaluate(test_loader)

        round_history.append({
            "round": r,
            "test_loss": round(loss, 4),
            "test_accuracy_pct": round(acc * 100.0, 2),
            "verification_passed": res["verification_passed"],
            "inspected_chunks": res["verify_info"]["inspected_chunks"]
        })

    return {
        "num_devices": num_devices,
        "rounds": rounds,
        "final_accuracy_pct": round_history[-1]["test_accuracy_pct"],
        "round_history": round_history
    }


if __name__ == "__main__":
    res = evaluate_nbaiot_iot_intrusion()
    print(json.dumps(res, indent=2))
