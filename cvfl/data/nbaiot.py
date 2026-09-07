import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple

class NBaIoTDeviceDataset(Dataset):
    """
    N-BaIoT (Network-based Detection of IoT Botnet Attacks) Dataset.
    Features: 115 statistical traffic features across 5 time windows (100ms, 500ms, 1.5s, 10s, 1min).
    Label: 0 (Benign IoT traffic), 1 (Botnet attack traffic - Mirai / Gafgyt).
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def generate_simulated_nbaiot_devices(
    num_devices: int = 9,
    samples_per_device: int = 600,
    feature_dim: int = 115
) -> Tuple[List[Dataset], Dataset]:
    """
    Simulates N-BaIoT dataset across N IoT devices with device-specific statistical distributions
    modelling real IoT device network behavior (non-IID by physical device type).
    
    Device types simulated:
    1. Danmini Doorbell
    2. Ecobee Thermostat
    3. Envision Security Cam
    4. Provision PTZ Cam
    5. Provision IR Cam
    6. Philips Baby Monitor
    7. SimpleHome Security Cam
    8. SimpleHome Doorbell
    9. SimpleHome Plug
    """
    device_datasets = []
    all_test_data = []
    all_test_labels = []

    np.random.seed(42)

    for dev_id in range(num_devices):
        # Base feature distribution unique to IoT device hardware profile
        device_mean = np.random.uniform(-1.5, 1.5, size=feature_dim)
        device_scale = np.random.uniform(0.5, 2.0, size=feature_dim)

        # Benign traffic (70%)
        num_benign = int(samples_per_device * 0.7)
        benign_X = np.random.normal(device_mean, device_scale, size=(num_benign, feature_dim))
        benign_y = np.zeros(num_benign, dtype=np.int64)

        # Botnet Attack traffic (30% - Mirai/Gafgyt flood signature)
        num_attack = samples_per_device - num_benign
        attack_shift = np.random.uniform(2.0, 4.0, size=feature_dim)
        attack_X = np.random.normal(device_mean + attack_shift, device_scale * 1.5, size=(num_attack, feature_dim))
        attack_y = np.ones(num_attack, dtype=np.int64)

        X = np.vstack([benign_X, attack_X])
        y = np.hstack([benign_y, attack_y])

        # Shuffle device dataset
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        # Train (80%) / Test (20%) split per device
        split = int(0.8 * len(X))
        train_X, test_X = X[:split], X[split:]
        train_y, test_y = y[:split], y[split:]

        device_datasets.append(NBaIoTDeviceDataset(train_X, train_y))
        all_test_data.append(test_X)
        all_test_labels.append(test_y)

    global_test_X = np.vstack(all_test_data)
    global_test_y = np.hstack(all_test_labels)
    global_test_dataset = NBaIoTDeviceDataset(global_test_X, global_test_y)

    return device_datasets, global_test_dataset
