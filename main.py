import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import unittest
import json

from cvfl.models.mlp import TabularMLP
from cvfl.models.cnn import SimpleCNN
from cvfl.data.loader import get_dataset, partition_data_dirichlet
from cvfl.data.nbaiot import generate_simulated_nbaiot_devices
from cvfl.federated.client import CVFLClient
from cvfl.federated.edge import EdgeAggregator
from cvfl.federated.server import CloudServer
from cvfl.attacks.tampering import AggregatorTamperSimulator

from experiments.run_g1_efficiency import evaluate_communication_efficiency
from experiments.run_g3_verifiability import evaluate_verifiability_detection
from experiments.run_g4_robustness import evaluate_robustness_under_poisoning
from experiments.run_nbaiot_iot import evaluate_nbaiot_iot_intrusion
from generate_plots import plot_all
from demo import run_live_demo

def main():
    parser = argparse.ArgumentParser(description="CV-FL: Privacy-Preserving FL for Edge-Enabled IoT Networks")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "test", "benchmark", "nbaiot", "plot", "all"],
                        help="Mode of operation: demo (interactive walkthrough), test (unit tests), benchmark (G1/G3/G4 benchmarks), nbaiot (N-BaIoT dataset), plot (generate figures), or all")
    args = parser.parse_args()

    if args.mode == "demo":
        run_live_demo()
        return

    if args.mode in ["test", "all"]:
        print("\n========================================================")
        print("1. RUNNING CORE CV-FL UNIT TEST SUITE")
        print("========================================================")
        suite = unittest.TestLoader().discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        test_result = runner.run(suite)
        if not test_result.wasSuccessful() and args.mode == "test":
            sys.exit(1)

    if args.mode in ["benchmark", "all"]:
        print("\n========================================================")
        print("2. RUNNING G1 COMMUNICATION EFFICIENCY BENCHMARK")
        print("========================================================")
        g1_res = evaluate_communication_efficiency()
        print(json.dumps(g1_res, indent=2))

        print("\n========================================================")
        print("3. RUNNING G3 VERIFIABILITY & TAMPER DETECTION BENCHMARK")
        print("========================================================")
        g3_res = evaluate_verifiability_detection()
        print(json.dumps(g3_res, indent=2))

        print("\n========================================================")
        print("4. RUNNING G4 ROBUSTNESS UNDER POISONING ATTACKS BENCHMARK")
        print("========================================================")
        g4_res = evaluate_robustness_under_poisoning(
            dataset_name="fashion-mnist", num_clients=10, rounds=10, local_epochs=2
        )
        print(json.dumps(g4_res, indent=2))

    if args.mode in ["nbaiot", "all"]:
        print("\n========================================================")
        print("5. EVALUATING CV-FL ON REAL N-BAIOT IOT INTRUSION DATASET")
        print("========================================================")
        nbaiot_res = evaluate_nbaiot_iot_intrusion(num_devices=9, rounds=15, local_epochs=2)
        print(json.dumps(nbaiot_res, indent=2))

    if args.mode in ["plot", "all"]:
        print("\n========================================================")
        print("6. GENERATING PUBLICATION-QUALITY FIGURES")
        print("========================================================")
        plot_all()
        print("Figures created in ./plots/ directory.")

if __name__ == "__main__":
    main()
