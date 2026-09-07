import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from cvfl.core.quantization import SuperIncreasingQuantizer
from cvfl.core.privacy import DualLayerMasking
from cvfl.core.commitment import CommitmentManager, MerkleTree
from cvfl.attacks.tampering import AggregatorTamperSimulator


class CloudServer:
    """
    Cloud Server (CS) Node - FedAvg aggregation with sign-based verification.

    Model updates use weighted FedAvg over locally trained client weights
    (SignScore weights from the edge aggregator). Sign vectors remain in the
    protocol for compression, privacy masking, and Merkle spot-check verification.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.1,
        chunk_size: int = 10,
        commitment_chunk_size: int = 64,
        spot_check_ratio: float = 0.1,
        lr_decay: float = 1.0,
        tamper_simulator: Optional[AggregatorTamperSimulator] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.base_learning_rate = learning_rate
        self.quantizer = SuperIncreasingQuantizer(chunk_size=chunk_size)
        self.privacy = DualLayerMasking()
        self.commitment_mgr = CommitmentManager(chunk_size=commitment_chunk_size)
        self.spot_check_ratio = spot_check_ratio
        self.lr_decay = lr_decay
        self.round_num = 0
        self.tamper_simulator = tamper_simulator

    def process_round(
        self,
        edge_masked_agg: np.ndarray,
        client_r1_masks: List[np.ndarray],
        weights: np.ndarray,
        client_commitments: List[str],
        merkle_trees: List[MerkleTree],
        original_d: int,
        client_signs: Optional[List[np.ndarray]] = None,
        client_local_weights: Optional[List[List[torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """
        Executes cloud aggregation, verifiability check, and global model update.
        """
        self.round_num += 1

        if client_signs is not None:
            stacked = np.stack(client_signs, axis=0)
            weighted_sum = np.sum(stacked * weights[:, np.newaxis], axis=0)
            agg_signs = np.sign(weighted_sum).astype(np.int8)
        else:
            agg_signs = (edge_masked_agg.astype(np.int8) - 1).astype(np.int8)

        if self.tamper_simulator:
            agg_signs = self.tamper_simulator.tamper_aggregate_signs(agg_signs)

        verification_passed, verify_info = self.commitment_mgr.spot_check_verify(
            client_commitments=client_commitments,
            merkle_trees=merkle_trees,
            reported_aggregate_signs=agg_signs,
            weights=weights,
            spot_check_ratio=self.spot_check_ratio
        )

        if not verification_passed:
            return {
                "success": False,
                "verification_passed": False,
                "verify_info": verify_info,
                "reported_signs": agg_signs
            }

        if client_local_weights is not None:
            self._apply_fedavg_weight_update(client_local_weights, weights)
        else:
            current_lr = self.base_learning_rate * (self.lr_decay ** (self.round_num - 1))
            self._apply_sign_fallback_update(agg_signs, lr=current_lr)

        return {
            "success": True,
            "verification_passed": True,
            "verify_info": verify_info,
            "reported_signs": agg_signs,
            "aggregation_mode": "fedavg" if client_local_weights is not None else "sign_fallback"
        }

    def _apply_fedavg_weight_update(
        self,
        client_local_weights: List[List[torch.Tensor]],
        weights: np.ndarray
    ):
        """Weighted FedAvg: W_global = sum_i(w_i * W_local_i)."""
        with torch.no_grad():
            for param_idx, global_param in enumerate(self.model.parameters()):
                aggregated = torch.zeros_like(global_param)
                for client_params, weight in zip(client_local_weights, weights):
                    aggregated.add_(client_params[param_idx], alpha=float(weight))
                global_param.copy_(aggregated)

    def _apply_sign_fallback_update(self, agg_signs: np.ndarray, lr: float):
        """Fallback sign-based update when local weights are unavailable."""
        curr_idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                param_signs = agg_signs[curr_idx: curr_idx + numel].reshape(param.shape)
                update_tensor = torch.tensor(
                    param_signs, dtype=param.dtype, device=param.device
                )
                param.add_(lr * update_tensor)
                curr_idx += numel

    def evaluate(self, test_loader, device: str = "cpu") -> Tuple[float, float]:
        self.model.to(device)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                total_loss += criterion(output, target).item() * len(target)
                correct += (output.argmax(dim=1) == target).sum().item()
                total += len(target)
        return (total_loss / total if total > 0 else 0.0,
                correct / total if total > 0 else 0.0)
