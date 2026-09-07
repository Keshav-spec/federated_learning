import hashlib
import os
import numpy as np
from typing import List, Tuple, Dict, Any

class MerkleTree:
    """
    Merkle Tree for fine-grained chunk-level commitments over sign vectors.
    Supports inclusion proof generation and verification for spot-checking.
    """

    def __init__(self, chunks: List[bytes], nonces: List[bytes] = None):
        self.chunks = chunks
        self.nonces = nonces if nonces is not None else [os.urandom(16) for _ in chunks]
        self.leaves = [self._hash_leaf(chunk, nonce) for chunk, nonce in zip(self.chunks, self.nonces)]
        self.tree = self._build_tree(self.leaves)

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _hash_leaf(self, chunk: bytes, nonce: bytes) -> str:
        return self._hash_bytes(chunk + b"||" + nonce)

    def _build_tree(self, leaves: List[str]) -> List[List[str]]:
        tree = [leaves]
        while len(tree[-1]) > 1:
            current_level = tree[-1]
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = self._hash_bytes((left + right).encode('utf-8'))
                next_level.append(combined)
            tree.append(next_level)
        return tree

    def get_root(self) -> str:
        return self.tree[-1][0] if self.tree else ""

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Generate Merkle proof path for leaf at index.
        Returns list of (sibling_hash, direction) where direction is 'L' or 'R'.
        """
        proof = []
        curr_idx = index
        for level in range(len(self.tree) - 1):
            current_level = self.tree[level]
            is_right = (curr_idx % 2 == 1)
            sibling_idx = curr_idx - 1 if is_right else curr_idx + 1
            if sibling_idx < len(current_level):
                sibling_hash = current_level[sibling_idx]
                proof.append((sibling_hash, 'L' if is_right else 'R'))
            else:
                proof.append((current_level[curr_idx], 'R'))
            curr_idx //= 2
        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Tuple[str, str]], root_hash: str) -> bool:
        curr_hash = leaf_hash
        for sibling_hash, direction in proof:
            if direction == 'L':
                combined = (sibling_hash + curr_hash).encode('utf-8')
            else:
                combined = (curr_hash + sibling_hash).encode('utf-8')
            curr_hash = hashlib.sha256(combined).hexdigest()
        return curr_hash == root_hash


class CommitmentManager:
    """
    Manages Client Commitments and Spot-Check Verification Protocol (G3).
    """

    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size

    def create_commitment(self, signs: np.ndarray) -> Tuple[str, MerkleTree, List[bytes]]:
        """
        Create Merkle commitment over client's pre-compression sign vector.
        """
        signs_bytes = signs.astype(np.int8).tobytes()
        num_chunks = int(np.ceil(len(signs_bytes) / self.chunk_size))
        chunks = [
            signs_bytes[i * self.chunk_size : (i + 1) * self.chunk_size]
            for i in range(num_chunks)
        ]
        nonces = [os.urandom(16) for _ in chunks]
        merkle_tree = MerkleTree(chunks, nonces)
        root = merkle_tree.get_root()
        return root, merkle_tree, nonces

    def spot_check_verify(
        self,
        client_commitments: List[str],
        merkle_trees: List[MerkleTree],
        reported_aggregate_signs: np.ndarray,
        weights: np.ndarray = None,
        spot_check_ratio: float = 0.1,
        tamper_injector=None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute spot-check verification protocol.
        Auditor samples k out of C chunk indices and checks:
        1) Merkle proof validity against published client commitments.
        2) Recomputed chunk aggregate vs reported aggregate.
        """
        if not merkle_trees:
            return True, {"passed": True, "inspected_chunks": 0, "false_positive": False}

        num_clients = len(merkle_trees)
        if weights is None:
            weights = np.ones(num_clients, dtype=np.float64) / num_clients

        num_chunks = len(merkle_trees[0].chunks)
        k = max(1, int(np.ceil(num_chunks * spot_check_ratio)))
        selected_chunk_indices = np.random.choice(num_chunks, size=k, replace=False)

        signs_bytes = reported_aggregate_signs.astype(np.int8).tobytes()
        reported_chunks = [
            signs_bytes[i * self.chunk_size : (i + 1) * self.chunk_size]
            for i in range(num_chunks)
        ]

        verified_chunks = 0
        for chunk_idx in selected_chunk_indices:
            # 1. Verify Merkle proof for each client on chunk_idx
            recomputed_chunk_sum = None
            
            for client_idx, (commitment, tree) in enumerate(zip(client_commitments, merkle_trees)):
                chunk_data = tree.chunks[chunk_idx]
                nonce = tree.nonces[chunk_idx]
                leaf_hash = tree._hash_leaf(chunk_data, nonce)
                proof = tree.get_proof(chunk_idx)
                
                if not MerkleTree.verify_proof(leaf_hash, proof, commitment):
                    return False, {
                        "passed": False,
                        "failed_reason": f"Merkle proof invalid for client {client_idx} on chunk {chunk_idx}",
                        "inspected_chunks": verified_chunks
                    }
                
                client_signs_chunk = np.frombuffer(chunk_data, dtype=np.int8).astype(np.float64)
                w_i = weights[client_idx]
                if recomputed_chunk_sum is None:
                    recomputed_chunk_sum = client_signs_chunk * w_i
                else:
                    recomputed_chunk_sum += client_signs_chunk * w_i

            # Recomputed aggregated signs for chunk
            expected_agg_chunk_signs = np.sign(recomputed_chunk_sum).astype(np.int8)
            reported_chunk_signs = np.frombuffer(reported_chunks[chunk_idx], dtype=np.int8)
            
            # Check consistency
            if not np.array_equal(expected_agg_chunk_signs, reported_chunk_signs[:len(expected_agg_chunk_signs)]):
                return False, {
                    "passed": False,
                    "failed_reason": f"Aggregate mismatch on chunk {chunk_idx}",
                    "inspected_chunks": verified_chunks
                }
            
            verified_chunks += 1

        return True, {"passed": True, "inspected_chunks": verified_chunks}


class PedersenCommitment:
    """
    Homomorphic Pedersen Commitment helper over modular integers.
    C_i = (g^m_i * h^r_i) mod p
    Product(C_i) mod p = (g^Sum(m_i) * h^Sum(r_i)) mod p
    """
    def __init__(self, p: int = 2**31 - 1, g: int = 3, h: int = 7):
        self.p = p
        self.g = g
        self.h = h

    def commit(self, val: int, nonce: int) -> int:
        return (pow(self.g, val, self.p) * pow(self.h, nonce, self.p)) % self.p

    def verify_aggregate(self, commitments: List[int], total_val: int, total_nonce: int) -> bool:
        prod_commitments = 1
        for c in commitments:
            prod_commitments = (prod_commitments * c) % self.p
        expected = self.commit(total_val, total_nonce)
        return prod_commitments == expected
