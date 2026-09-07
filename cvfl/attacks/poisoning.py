import torch
import numpy as np
from typing import Tuple

class PoisoningAttacker:
    """
    Simulates malicious client behavior (Poisoning Attacks for G4 evaluation).
    """

    def __init__(self, attack_type: str = "label_flip", attack_scale: float = 1.0):
        """
        :param attack_type: 'label_flip', 'sign_flip', 'gaussian_noise', or 'backdoor'
        :param attack_scale: Scaling factor for noise/sign-flip
        """
        self.attack_type = attack_type.lower()
        self.attack_scale = attack_scale

    def poison_labels(self, data: torch.Tensor, target: torch.Tensor, num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply label-flipping attack.
        """
        if self.attack_type == "label_flip":
            poisoned_target = (num_classes - 1) - target
            return data, poisoned_target
        elif self.attack_type == "backdoor":
            # Add small trigger box at bottom-right pixel corner
            poisoned_data = data.clone()
            if data.dim() == 4: # Image tensor (B, C, H, W)
                poisoned_data[:, :, -2:, -2:] = 1.0
            elif data.dim() == 2: # Tabular feature tensor (B, D)
                poisoned_data[:, -5:] = 5.0
            poisoned_target = torch.zeros_like(target) # Target class 0
            return poisoned_data, poisoned_target
        return data, target

    def poison_signs(self, sign_vector: np.ndarray) -> np.ndarray:
        """
        Apply gradient-level poisoning (sign-flipping or noise).
        """
        if self.attack_type == "sign_flip":
            return (-self.attack_scale * sign_vector).astype(np.int8)
        elif self.attack_type == "gaussian_noise":
            noise = np.random.normal(0, self.attack_scale, size=sign_vector.shape)
            noisy_signs = np.sign(sign_vector + noise).astype(np.int8)
            return noisy_signs
        return sign_vector
