import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_g1_efficiency import evaluate_communication_efficiency
from experiments.run_g3_verifiability import evaluate_verifiability_detection
from experiments.run_g4_robustness import evaluate_robustness_under_poisoning
from experiments.run_nbaiot_iot import evaluate_nbaiot_iot_intrusion

def plot_all():
    os.makedirs("./plots", exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: G1 Communication Efficiency
    print("Generating Figure 1: G1 Communication Bandwidth Comparison...")
    g1_data = evaluate_communication_efficiency()
    models = list(g1_data.keys())
    
    fedavg_bytes = [g1_data[m]["fedavg_bytes"] / 1024.0 for m in models]
    sign_fl_bytes = [g1_data[m]["sign_fl_bytes"] / 1024.0 for m in models]
    cvfl_bytes = [g1_data[m]["cvfl_bytes"] / 1024.0 for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects1 = ax.bar(x - width, fedavg_bytes, width, label='FedAvg (Uncompressed)', color='#d9534f')
    rects2 = ax.bar(x, sign_fl_bytes, width, label='Sign-FL (Paper 4)', color='#f0ad4e')
    rects3 = ax.bar(x + width, cvfl_bytes, width, label='CV-FL (Proposed Method)', color='#5cb85c')

    ax.set_ylabel('Payload Size per Update (KB)', fontsize=12, fontweight='bold')
    ax.set_title('G1: Communication Efficiency Comparison (Upload Bytes / Client / Round)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.split()[0] for m in models], fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig("./plots/g1_bandwidth_comparison.png")
    plt.close()

    # Figure 2: G3 Verifiability Detection Curves
    print("Generating Figure 2: G3 Verifiability Detection Rate Curves...")
    g3_data = evaluate_verifiability_detection()
    ratios = [r * 100 for r in g3_data["spot_check_ratios"]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = ['#0275d8', '#5cb85c', '#f0ad4e', '#d9534f']
    markers = ['o', 's', '^', 'D']

    for idx, (tamper_key, curves) in enumerate(g3_data["detection_curves"].items()):
        label_pct = tamper_key.replace("tamper_", "").replace("pc", "% EA Tampering")
        ax.plot(ratios, [c * 100 for c in curves], label=label_pct, color=colors[idx], marker=markers[idx], linewidth=2.5)

    ax.set_xlabel('Spot-Check Sampling Ratio (k/d %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tamper Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('G3: Lightweight Verifiable Aggregation Detection Probability vs. Spot-Check Ratio', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig("./plots/g3_verifiability_detection_curve.png")
    plt.close()

    # Figure 3: G4 Robustness under Poisoning Attacks
    print("Generating Figure 3: G4 Robustness under Poisoning Attacks...")
    g4_data = evaluate_robustness_under_poisoning(dataset_name="fashion-mnist", num_clients=10, rounds=5)
    mal_ratios = [r * 100 for r in g4_data["malicious_ratios"]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    attack_colors = {'label_flip': '#d9534f', 'sign_flip': '#f0ad4e', 'gaussian_noise': '#0275d8'}
    attack_markers = {'label_flip': 'o', 'sign_flip': 's', 'gaussian_noise': '^'}

    for attack, results_dict in g4_data["experiments"].items():
        accs = list(results_dict.values())
        ax.plot(mal_ratios, accs, label=f"Attack: {attack.replace('_', ' ').title()}", color=attack_colors[attack], marker=attack_markers[attack], linewidth=2.5)

    ax.set_xlabel('Malicious Client Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Model Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('G4: CV-FL Robustness against Poisoning Attacks (Non-IID Dirichlet α=0.5)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("./plots/g4_robustness_poisoning_attacks.png")
    plt.close()

    # Figure 4: N-BaIoT IoT Intrusion Training
    print("Generating Figure 4: N-BaIoT IoT Intrusion Detection Training...")
    nbaiot_data = evaluate_nbaiot_iot_intrusion(num_devices=9, rounds=10)
    rounds = [r["round"] for r in nbaiot_data["round_history"]]
    accs = [r["test_accuracy_pct"] for r in nbaiot_data["round_history"]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(rounds, accs, color='#5cb85c', marker='o', linewidth=2.5, label='CV-FL (9 IoT Device Nodes)')
    ax.set_xlabel('Federated Communication Rounds', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Intrusion Detection Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('IoT Intrusion Detection Accuracy on N-BaIoT Dataset (Non-IID Devices)', fontsize=13, fontweight='bold')
    ax.set_ylim(50, 100)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("./plots/nbaiot_iot_intrusion_acc.png")
    plt.close()

    print("All plots generated successfully in ./plots/ directory!")

if __name__ == "__main__":
    plot_all()
