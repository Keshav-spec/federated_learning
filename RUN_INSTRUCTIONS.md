# Run Instructions for CV-FL (Privacy-Preserving Federated Learning for Edge-Enabled IoT Networks)

## Prerequisites
- **Python 3.9+** (tested on Windows 10/11 with Python 3.10, 3.11, 3.12)
- **Virtual Environment** (recommended)

---

## Step 1: Virtual Environment Setup
From the project root (`c:\Users\vinay\Desktop\7th sem\predictive\Projec`):
```powershell
# Create virtual environment (if not already created)
python -m venv venv

# Activate on PowerShell
.\venv\Scripts\Activate.ps1

# Or on Command Prompt (CMD)
venv\Scripts\activate.bat
```

---

## Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```
*Note: If running on a system without GPU/CUDA, standard CPU PyTorch is automatically used.*

---

## Step 3: Run the System

### Option A: Interactive Live Demonstration
Runs a complete walkthrough of Stages 0 to 4, Byzantine poisoning defense, Dishonest Aggregator tamper detection, and multi-round convergence on N-BaIoT IoT telemetry:
```powershell
# Direct Python execution:
python demo.py

# Or using the Windows Batch launcher:
.\run_demo.bat

# Or using the PowerShell launcher:
.\run_demo.ps1
```

### Option B: Unified CLI Runner (`main.py`)
You can run specific modules or the full experimental evaluation pipeline via `main.py`:

| Command | Description |
|---|---|
| `python main.py --mode demo` | Launches the interactive CV-FL demonstration |
| `python main.py --mode test` | Executes the 9-point unit test suite across all cryptographic and ML modules |
| `python main.py --mode benchmark` | Runs G1 (Efficiency), G3 (Verifiability), and G4 (Robustness) benchmarks |
| `python main.py --mode nbaiot` | Evaluates multi-round IoT intrusion detection across 9 non-IID device nodes |
| `python main.py --mode plot` | Generates 4 publication-quality figures saved to `./plots/` |
| `python main.py --mode all` | Runs tests, all benchmarks, N-BaIoT evaluation, and generates all figures |

### Option C: Standalone Experiment Scripts
Each experiment can be executed independently:
```powershell
# G1 Communication Bandwidth & Compression Ratio
python experiments/run_g1_efficiency.py

# G3 Merkle Spot-Check Verifiability & Tamper Catch Rates
python experiments/run_g3_verifiability.py

# G4 Byzantine Robustness under Label-Flip, Sign-Flip, and Gaussian Noise
python experiments/run_g4_robustness.py

# Real Non-IID N-BaIoT IoT Device Telemetry Evaluation
python experiments/run_nbaiot_iot.py

# Generate High-Resolution Publication Plots
python generate_plots.py
```

---

## Generated Visualizations (`./plots/`)
1. **`g1_bandwidth_comparison.png`**: Client upload cost comparison (FedAvg vs Sign-Based FL vs CV-FL) showing ~19-32x compression.
2. **`g3_verifiability_detection_curve.png`**: Spot-check detection probability vs sampling ratio ($k/d$) under 1%-20% EA tampering.
3. **`g4_robustness_poisoning_attacks.png`**: Global accuracy under varying malicious client ratios (0%-40%) for Label-Flip, Sign-Flip, and Gaussian Noise attacks.
4. **`nbaiot_iot_intrusion_acc.png`**: Multi-round convergence curve on real N-BaIoT IoT botnet telemetry.

