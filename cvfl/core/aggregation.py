import numpy as np
from typing import List, Tuple

class SignScoreAggregator:
    """
    SignScore Robust Aggregation (G4).
    Calculates element-wise sign agreement against the coordinate-wise median,
    down-weighting statistically anomalous / malicious client updates.
    """

    def __init__(self, temperature: float = 5.0):
        self.temperature = temperature

    def compute_median_sign(self, client_signs: List[np.ndarray]) -> np.ndarray:
        """
        Compute coordinate-wise median sign vector across all clients.
        """
        stacked = np.stack(client_signs, axis=0) # Shape: (N, d)
        median_vals = np.median(stacked, axis=0)
        median_signs = np.sign(median_vals).astype(np.int8)
        return median_signs

    def compute_sign_scores(self, client_signs: List[np.ndarray], median_sign: np.ndarray) -> np.ndarray:
        """
        Compute SignScore S_i for each client: fraction of matching dimensions with median sign.
        S_i = (1 / d) * sum_j I(q_{i,j} == m_j)
        """
        scores = []
        for signs in client_signs:
            agreement = np.equal(signs, median_sign)
            score = np.mean(agreement)
            scores.append(score)
        return np.array(scores, dtype=np.float64)

    def compute_robust_weights(self, sign_scores: np.ndarray, adaptive_filter: bool = True) -> np.ndarray:
        """
        Compute normalized robust weights via softmax over distance D_i = 1 - S_i.
        With adaptive_filter=True, statistically anomalous updates (malicious outliers)
        are dynamically identified and zeroed out before softmax normalization.
        """
        weights = np.zeros_like(sign_scores, dtype=np.float64)
        if adaptive_filter and len(sign_scores) > 2:
            median_score = float(np.median(sign_scores))
            valid_mask = sign_scores >= max(0.40, median_score - 0.15)
            if not np.any(valid_mask):
                valid_mask = np.ones(len(sign_scores), dtype=bool)
        else:
            valid_mask = np.ones(len(sign_scores), dtype=bool)

        distances = 1.0 - sign_scores[valid_mask]
        exp_neg = np.exp(-self.temperature * distances)
        norm_w = exp_neg / (np.sum(exp_neg) + 1e-12)
        weights[valid_mask] = norm_w
        return weights

    def aggregate(self, client_signs: List[np.ndarray], adaptive_filter: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform robust sign aggregation.
        Returns: (aggregated_sign_vector, sign_scores, robust_weights)
        """
        median_sign = self.compute_median_sign(client_signs)
        sign_scores = self.compute_sign_scores(client_signs, median_sign)
        weights = self.compute_robust_weights(sign_scores, adaptive_filter=adaptive_filter)
        
        stacked = np.stack(client_signs, axis=0)
        # Weighted sum of sign vectors
        weighted_sum = np.sum(stacked * weights[:, np.newaxis], axis=0)
        agg_signs = np.sign(weighted_sum).astype(np.int8)
        return agg_signs, sign_scores, weights
