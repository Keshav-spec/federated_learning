import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for Tabular IoT Data (N-BaIoT) and tabularized images (MNIST).
    """

    def __init__(self, input_dim: int = 115, hidden_dim: int = 64, num_classes: int = 2):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        logits = self.out(x)
        return logits
