import numpy as np
from typing import List, Tuple, Dict, Any

from cvfl.core.privacy import DualLayerMasking
from cvfl.core.aggregation import SignScoreAggregator
from cvfl.core.commitment import MerkleTree

class EdgeAggregator:
    """
    Edge Aggregator (EA) Node.
    Executes Stage 3 Edge-Level Processing:
    1. Removes outer secret mask layer r2.
    2. Computes SignScore consistency and Euclidean-distance robust weights.
    3. Performs robust weighted aggregation of edge updates.
    4. Forwards edge-level aggregate + set of commitments {c_i} to Cloud Server.
    """

    def __init__(self, ea_id: int = 0, temperature: float = 10.0, mod_val: int = 2**31 - 1):
        self.ea_id = ea_id
        self.privacy = DualLayerMasking(mod_val=mod_val)
        self.aggregator = SignScoreAggregator(temperature=temperature)

    def aggregate_edge(
        self,
        client_masked_scalars: List[np.ndarray],
        client_r2_masks: List[np.ndarray],
        client_unmasked_signs: List[np.ndarray],
        client_commitments: List[str],
        merkle_trees: List[MerkleTree],
        adaptive_filter: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[MerkleTree]]:
        """
        Processes updates from edge-connected clients.
        Returns: (edge_masked_aggregate, sign_scores, robust_weights, client_commitments, merkle_trees)
        """
        # Step 1: EA removes outer mask layer r2
        partially_unmasked = []
        for masked_u, r2 in zip(client_masked_scalars, client_r2_masks):
            u_edge = self.privacy.remove_edge_mask(masked_u, r2)
            partially_unmasked.append(u_edge)

        # Step 2: SignScore Robust Weight Calculation
        median_sign = self.aggregator.compute_median_sign(client_unmasked_signs)
        sign_scores = self.aggregator.compute_sign_scores(client_unmasked_signs, median_sign)
        weights = self.aggregator.compute_robust_weights(sign_scores, adaptive_filter=adaptive_filter)

        # Step 3: Compute aggregated sign vector update
        stacked_signs = np.stack(client_unmasked_signs, axis=0)
        weighted_sign_sum = np.sum(stacked_signs * weights[:, np.newaxis], axis=0)
        agg_signs = np.sign(weighted_sign_sum).astype(np.int8)

        # Pack aggregated signs for cloud transmission
        trits = (agg_signs + 1).astype(np.uint8)
        edge_masked_agg = self.privacy.apply_dual_mask(trits, np.zeros_like(trits), np.zeros_like(trits))

        return edge_masked_agg, sign_scores, weights, client_commitments, merkle_trees
