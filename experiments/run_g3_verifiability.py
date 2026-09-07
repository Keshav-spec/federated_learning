import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import time
import json
from typing import Dict, Any

from cvfl.core.commitment import CommitmentManager
from cvfl.attacks.tampering import AggregatorTamperSimulator

def evaluate_verifiability_detection() -> Dict[str, Any]:
    """
    Evaluates G3 Verifiability: Detection Rate vs Spot-Check Ratio (k/d) and Runtime Overhead.
    Fast execution.
    """
    commitment_mgr = CommitmentManager(chunk_size=64)
    spot_check_ratios = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    tamper_rates = [0.01, 0.05, 0.10, 0.20]
    num_trials = 10

    d = 2000 # 2,000 dimensions model sign update
    num_clients = 5

    results = {"spot_check_ratios": spot_check_ratios, "detection_curves": {}, "verification_runtimes_ms": {}}

    for tamper_rate in tamper_rates:
        detection_rates = []
        runtimes = []

        tamper_sim = AggregatorTamperSimulator(tamper_rate=tamper_rate, tamper_mode="sign_flip")

        for ratio in spot_check_ratios:
            detected_count = 0
            total_time_ms = 0.0

            for trial in range(num_trials):
                np.random.seed(trial + int(ratio * 1000))
                
                client_signs = [
                    np.random.choice([-1, 0, 1], size=d).astype(np.int8)
                    for _ in range(num_clients)
                ]

                client_commitments = []
                merkle_trees = []
                for signs in client_signs:
                    root, tree, _ = commitment_mgr.create_commitment(signs)
                    client_commitments.append(root)
                    merkle_trees.append(tree)

                stacked = np.stack(client_signs, axis=0)
                honest_agg_signs = np.sign(np.sum(stacked, axis=0)).astype(np.int8)
                tampered_agg_signs = tamper_sim.tamper_aggregate_signs(honest_agg_signs)

                start_t = time.time()
                passed, info = commitment_mgr.spot_check_verify(
                    client_commitments=client_commitments,
                    merkle_trees=merkle_trees,
                    reported_aggregate_signs=tampered_agg_signs,
                    spot_check_ratio=ratio
                )
                elapsed_ms = (time.time() - start_t) * 1000.0
                total_time_ms += elapsed_ms

                if not passed:
                    detected_count += 1

            detection_rate = detected_count / num_trials
            avg_runtime_ms = total_time_ms / num_trials

            detection_rates.append(round(detection_rate, 4))
            runtimes.append(round(avg_runtime_ms, 3))

        results["detection_curves"][f"tamper_{int(tamper_rate*100)}pc"] = detection_rates
        results["verification_runtimes_ms"][f"tamper_{int(tamper_rate*100)}pc"] = runtimes

    return results

if __name__ == "__main__":
    res = evaluate_verifiability_detection()
    print(json.dumps(res, indent=2))
