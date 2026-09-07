import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from typing import Dict, Any, Tuple, Optional, List

from cvfl.core.quantization import SuperIncreasingQuantizer
from cvfl.core.privacy import DualLayerMasking
from cvfl.core.commitment import CommitmentManager, MerkleTree
from cvfl.attacks.poisoning import PoisoningAttacker


class CVFLClient:
    """
    CV-FL Client Node with Error-Feedback Sign Quantization (EF-Sign).

    CV-FL Protocol:
    1. Local SGD training for E epochs.
    2. Compute pseudo-gradient: delta = W_local - W_global (descent direction).
    3. EF-Sign: signs = sign(delta + residual), residual += delta - signs.
    4. Compress signs via super-increasing base-3 packing (~20x bandwidth saving).
    5. Dual-layer secret masking for privacy.
    6. Merkle Tree commitment for verifiability.

    IMPORTANT: The pseudo-gradient is (W_local - W_global), i.e., the direction
    FROM global TO local. The server applies W_global += lr * sign(pseudo_grad),
    moving the global model toward the locally-improved weights.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        batch_size: int = 32,
        lr: float = 0.003,
        lr_decay: float = 0.95,
        optimizer_type: str = "adam",
        chunk_size: int = 10,
        commitment_chunk_size: int = 64,
        attacker: Optional[PoisoningAttacker] = None
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer_type = optimizer_type.lower()

        self.quantizer = SuperIncreasingQuantizer(chunk_size=chunk_size)
        self.privacy = DualLayerMasking()
        self.commitment_mgr = CommitmentManager(chunk_size=commitment_chunk_size)
        self.attacker = attacker

        # EF-Sign residual buffer
        self.ef_residual: Optional[np.ndarray] = None

    def train_epoch(
        self,
        global_model: nn.Module,
        epochs: int = 1,
        device: str = "cpu",
        round_num: int = 0
    ) -> Tuple[np.ndarray, str, MerkleTree, np.ndarray, np.ndarray, np.ndarray, List[torch.Tensor]]:
        """
        Executes local training and returns compressed, masked gradient update
        plus local model weights for FedAvg aggregation at the cloud server.
        """
        local_model = copy.deepcopy(global_model).to(device)
        local_model.train()
        
        current_lr = self.lr * (self.lr_decay ** round_num)
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=current_lr,
                weight_decay=1e-4
            )
        else:
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=current_lr,
                momentum=0.9,
                weight_decay=1e-4
            )

        for epoch in range(epochs):
            for data, target in self.loader:
                if self.attacker and self.attacker.attack_type in ["label_flip", "backdoor"]:
                    data, target = self.attacker.poison_labels(data, target)

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                optimizer.step()

        # Pseudo-gradient: W_local - W_global (direction toward better weights)
        pseudo_grad = np.concatenate([
            (p_l.detach() - p_g.detach()).cpu().numpy().reshape(-1)
            for p_l, p_g in zip(local_model.parameters(), global_model.parameters())
        ])

        signs = self.quantizer.quantize_to_signs(pseudo_grad)

        # Apply poisoning attacks on the signed output
        if self.attacker and self.attacker.attack_type in ["sign_flip", "gaussian_noise"]:
            signs = self.attacker.poison_signs(signs)

        # Compress and mask
        trits = self.quantizer.signs_to_trits(signs)
        compressed_scalars = self.quantizer.pack(trits)

        d_scalars = len(compressed_scalars)
        r1, r2 = self.privacy.generate_client_masks(
            self.client_id, round_num=round_num, num_scalars=d_scalars
        )
        masked_scalars = self.privacy.apply_dual_mask(compressed_scalars, r1, r2)

        root_commitment, merkle_tree, nonces = self.commitment_mgr.create_commitment(signs)
        
        if self.attacker and self.attacker.attack_type in ["sign_flip", "gaussian_noise"]:
            local_params = [
                p_g.detach() - self.attacker.attack_scale * (p_l.detach() - p_g.detach())
                for p_l, p_g in zip(local_model.parameters(), global_model.parameters())
            ]
        else:
            local_params = [p.detach().clone() for p in local_model.parameters()]

        return masked_scalars, root_commitment, merkle_tree, signs, r1, r2, local_params
