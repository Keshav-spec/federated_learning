import numpy as np

class DualLayerMasking:
    """
    Dual-Layer Secret Masking for Privacy Preservation (G2).
    Applies additive secret masks r1 (cloud-level) and r2 (edge-level) to compressed vectors.
    
    Protects client raw updates against non-colluding Edge Aggregator (EA) and Cloud Server (CS).
    - EA only sees (u_i + r1_i + r2_i) and removes r2_i -> cannot learn u_i due to r1_i mask.
    - CS receives edge aggregate with r1_i and removes r1_i -> cannot isolate individual u_i.
    """

    def __init__(self, mod_val: int = 2**31 - 1):
        """
        :param mod_val: Large prime modulus for additive secret sharing arithmetic.
        """
        self.mod_val = mod_val

    def generate_client_masks(self, client_id: int, round_num: int, num_scalars: int):
        """
        Derive pseudorandom noise vectors r1 (Cloud mask) and r2 (Edge mask) for client i in round.
        Uses deterministic seeds derived from (client_id, round_num) to model pairwise key agreements.
        """
        seed_r1 = hash((client_id, round_num, "cloud_r1")) & 0xFFFFFFFF
        seed_r2 = hash((client_id, round_num, "edge_r2")) & 0xFFFFFFFF
        
        rng_r1 = np.random.RandomState(seed_r1)
        rng_r2 = np.random.RandomState(seed_r2)

        r1 = rng_r1.randint(0, self.mod_val, size=num_scalars, dtype=np.int64)
        r2 = rng_r2.randint(0, self.mod_val, size=num_scalars, dtype=np.int64)
        return r1, r2

    def apply_dual_mask(self, u_i: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Client computes: \tilde{u}_i = (u_i + r1 + r2) mod M
        """
        masked = (u_i.astype(np.int64) + r1 + r2) % self.mod_val
        return masked

    def remove_edge_mask(self, masked_u_i: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        EA computes: \tilde{u}_i^{edge} = (\tilde{u}_i - r2) mod M = (u_i + r1) mod M
        """
        u_edge = (masked_u_i.astype(np.int64) - r2) % self.mod_val
        return u_edge

    def remove_cloud_mask_aggregate(self, masked_aggregate: np.ndarray, r1_aggregate: np.ndarray) -> np.ndarray:
        """
        CS computes: u_final = (masked_aggregate - r1_aggregate) mod M
        """
        unmasked = (masked_aggregate.astype(np.int64) - r1_aggregate) % self.mod_val
        return unmasked
