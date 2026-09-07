import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import json
from typing import Dict, Any

from cvfl.models.mlp import TabularMLP
from cvfl.models.cnn import SimpleCNN
from cvfl.core.quantization import SuperIncreasingQuantizer
from cvfl.core.commitment import CommitmentManager

def evaluate_communication_efficiency() -> Dict[str, Any]:
    """
    Evaluates G1 Communication Efficiency (Bytes uploaded per client per round).
    Compares:
    1. Standard FedAvg (32-bit float parameters)
    2. Sign-Based FL (Paper 4: super-increasing compressed sign vector)
    3. Pairing-Based VFE (Paper 5: uncompressed floats + 512-bit G1/G2 pairing signatures)
    4. CV-FL (Our Method: super-increasing compression + 256-bit SHA-256 Merkle root commitment)
    """
    models = {
        "TabularMLP (N-BaIoT / MNIST)": TabularMLP(input_dim=115, hidden_dim=64, num_classes=2),
        "SimpleCNN (Fashion-MNIST)": SimpleCNN(in_channels=1, num_classes=10),
        "SimpleCNN (CIFAR-10)": SimpleCNN(in_channels=3, num_classes=10)
    }

    results = {}

    for model_name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())

        # 1. FedAvg (32-bit floats = 4 bytes per param)
        fedavg_bytes = num_params * 4

        # 2. Sign-Based FL (Paper 4: 10 trits per packed 16-bit integer = ~1.6 bits per param)
        quantizer = SuperIncreasingQuantizer(chunk_size=10)
        num_scalars = int(np.ceil(num_params / 10))
        sign_fl_bytes = num_scalars * 2 # 16-bit int scalars

        # 3. Pairing VFE (32-bit floats + 64 bytes pairing proof)
        vfe_bytes = fedavg_bytes + 64

        # 4. CV-FL (Sign-Based FL compressed scalars + 32 bytes SHA-256 Merkle root commitment)
        cvfl_bytes = sign_fl_bytes + 32

        compression_ratio = fedavg_bytes / cvfl_bytes

        results[model_name] = {
            "num_parameters": num_params,
            "fedavg_bytes": fedavg_bytes,
            "sign_fl_bytes": sign_fl_bytes,
            "vfe_bytes": vfe_bytes,
            "cvfl_bytes": cvfl_bytes,
            "commitment_overhead_bytes": 32,
            "compression_ratio_vs_fedavg": round(compression_ratio, 2)
        }

    return results

if __name__ == "__main__":
    res = evaluate_communication_efficiency()
    print(json.dumps(res, indent=2))
