import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
import torch
import torch.nn as nn

from cvfl.core.quantization import SuperIncreasingQuantizer
from cvfl.core.privacy import DualLayerMasking
from cvfl.core.commitment import MerkleTree, CommitmentManager, PedersenCommitment
from cvfl.core.aggregation import SignScoreAggregator
from cvfl.models.mlp import TabularMLP
from cvfl.models.cnn import SimpleCNN
from cvfl.federated.client import CVFLClient
from cvfl.federated.edge import EdgeAggregator
from cvfl.federated.server import CloudServer
from cvfl.data.nbaiot import generate_simulated_nbaiot_devices
from cvfl.attacks.tampering import AggregatorTamperSimulator
from cvfl.attacks.poisoning import PoisoningAttacker
from cvfl.federated.flwr_strategy import CVFLStrategy

class TestCVFLCore(unittest.TestCase):

    def test_super_increasing_quantization_lossless(self):
        """
        Verify that sign vector quantization + super-increasing compression is 100% lossless.
        """
        quantizer = SuperIncreasingQuantizer(chunk_size=10)
        np.random.seed(42)
        original_signs = np.random.choice([-1, 0, 1], size=105).astype(np.int8)

        trits = quantizer.signs_to_trits(original_signs)
        packed = quantizer.pack(trits)
        unpacked_trits = quantizer.unpack(packed, len(original_signs))
        recovered_signs = quantizer.trits_to_signs(unpacked_trits)

        np.testing.assert_array_equal(original_signs, recovered_signs)

    def test_dual_layer_masking(self):
        """
        Verify that dual-layer secret masking (r1, r2) unmasks correctly across EA and CS.
        """
        privacy = DualLayerMasking(mod_val=2**31 - 1)
        u_i = np.array([10, 20, 30, 40, 50], dtype=np.int64)

        r1, r2 = privacy.generate_client_masks(client_id=1, round_num=1, num_scalars=len(u_i))
        masked_u_i = privacy.apply_dual_mask(u_i, r1, r2)

        # EA removes r2
        u_edge = privacy.remove_edge_mask(masked_u_i, r2)
        expected_u_edge = (u_i + r1) % privacy.mod_val
        np.testing.assert_array_equal(u_edge, expected_u_edge)

        # CS removes r1
        unmasked = privacy.remove_cloud_mask_aggregate(u_edge, r1)
        np.testing.assert_array_equal(unmasked, u_i)

    def test_merkle_tree_proofs_and_spot_check(self):
        """
        Verify Merkle tree root computation, inclusion proofs, and spot-check tamper detection.
        """
        commitment_mgr = CommitmentManager(chunk_size=16)
        np.random.seed(42)
        signs_client1 = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        signs_client2 = np.random.choice([-1, 0, 1], size=64).astype(np.int8)

        c1, tree1, nonces1 = commitment_mgr.create_commitment(signs_client1)
        c2, tree2, nonces2 = commitment_mgr.create_commitment(signs_client2)

        agg_signs = np.sign(signs_client1.astype(np.int32) + signs_client2.astype(np.int32)).astype(np.int8)

        # Honest verification should pass
        passed, info = commitment_mgr.spot_check_verify(
            client_commitments=[c1, c2],
            merkle_trees=[tree1, tree2],
            reported_aggregate_signs=agg_signs,
            spot_check_ratio=1.0 # 100% inspection for test
        )
        self.assertTrue(passed)

        # Tampered aggregate should fail verification
        tampered_signs = agg_signs.copy()
        tampered_signs[0] = -tampered_signs[0]
        passed_tamper, info_tamper = commitment_mgr.spot_check_verify(
            client_commitments=[c1, c2],
            merkle_trees=[tree1, tree2],
            reported_aggregate_signs=tampered_signs,
            spot_check_ratio=1.0
        )
        self.assertFalse(passed_tamper)

    def test_pedersen_homomorphic_commitment(self):
        """
        Verify homomorphic property of Pedersen commitment: Prod(C_i) = Commit(Sum(m_i), Sum(r_i)).
        """
        pedersen = PedersenCommitment(p=2**31 - 1, g=3, h=7)
        vals = [12, 45, 78]
        nonces = [101, 203, 305]

        commitments = [pedersen.commit(v, r) for v, r in zip(vals, nonces)]
        total_val = sum(vals)
        total_nonce = sum(nonces)

        self.assertTrue(pedersen.verify_aggregate(commitments, total_val, total_nonce))

    def test_sign_score_downweighting(self):
        """
        Verify SignScore correctly down-weights malicious/anomalous sign updates.
        """
        aggregator = SignScoreAggregator(temperature=5.0)
        honest_signs1 = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        honest_signs2 = np.array([1, 1, 1, 1, 1], dtype=np.int8)
        honest_signs3 = np.array([1, 1, 1, 0, 1], dtype=np.int8)
        malicious_signs = np.array([-1, -1, -1, -1, -1], dtype=np.int8) # Sign-flipped attacker

        client_signs = [honest_signs1, honest_signs2, honest_signs3, malicious_signs]
        agg_signs, scores, weights = aggregator.aggregate(client_signs)

        # Malicious client should have lowest weight
        self.assertLess(weights[3], weights[0])
        self.assertLess(weights[3], weights[1])

    def test_aggregator_tampering_modes(self):
        """
        Verify tamper simulator generates altered sign vectors across all modes.
        """
        original = np.ones(50, dtype=np.int8)
        
        sim_flip = AggregatorTamperSimulator(tamper_rate=0.2, tamper_mode="sign_flip")
        tampered_flip = sim_flip.tamper_aggregate_signs(original)
        self.assertFalse(np.array_equal(original, tampered_flip))

        sim_zero = AggregatorTamperSimulator(tamper_rate=0.2, tamper_mode="zero_out")
        tampered_zero = sim_zero.tamper_aggregate_signs(original)
        self.assertTrue(np.any(tampered_zero == 0))

        sim_rand = AggregatorTamperSimulator(tamper_rate=0.2, tamper_mode="random_noise")
        tampered_rand = sim_rand.tamper_aggregate_signs(original)
        self.assertFalse(np.array_equal(original, tampered_rand))

    def test_poisoning_attacker_modes(self):
        """
        Verify poisoning attacker behaves properly for label-flip, backdoor, sign-flip, gaussian noise.
        """
        data = torch.randn(4, 10)
        target = torch.tensor([0, 1, 2, 3])

        # Label flip
        att_lf = PoisoningAttacker(attack_type="label_flip")
        _, p_target = att_lf.poison_labels(data, target, num_classes=10)
        self.assertTrue(torch.equal(p_target, torch.tensor([9, 8, 7, 6])))

        # Sign flip
        att_sf = PoisoningAttacker(attack_type="sign_flip", attack_scale=1.0)
        signs = np.array([1, -1, 1, 0], dtype=np.int8)
        p_signs = att_sf.poison_signs(signs)
        np.testing.assert_array_equal(p_signs, np.array([-1, 1, -1, 0], dtype=np.int8))

    def test_end_to_end_cvfl_round(self):
        """
        Verify complete end-to-end CV-FL execution round.
        """
        device_datasets, test_dataset = generate_simulated_nbaiot_devices(num_devices=3, samples_per_device=100)
        model = TabularMLP(input_dim=115, hidden_dim=32, num_classes=2)

        client1 = CVFLClient(client_id=0, dataset=device_datasets[0])
        client2 = CVFLClient(client_id=1, dataset=device_datasets[1])

        ea = EdgeAggregator()
        server = CloudServer(model=model, spot_check_ratio=0.1)

        m1, c1, tree1, signs1, r1_1, r2_1, w1 = client1.train_epoch(model, epochs=1, round_num=0)
        m2, c2, tree2, signs2, r1_2, r2_2, w2 = client2.train_epoch(model, epochs=1, round_num=0)

        # EA executes edge stage
        edge_agg, scores, weights, commitments, merkle_trees = ea.aggregate_edge(
            client_masked_scalars=[m1, m2],
            client_r2_masks=[r2_1, r2_2],
            client_unmasked_signs=[signs1, signs2],
            client_commitments=[c1, c2],
            merkle_trees=[tree1, tree2]
        )

        # CS executes cloud stage
        res = server.process_round(
            edge_masked_agg=edge_agg,
            client_r1_masks=[r1_1, r1_2],
            weights=weights,
            client_commitments=commitments,
            merkle_trees=merkle_trees,
            original_d=len(signs1),
            client_signs=[signs1, signs2],
            client_local_weights=[w1, w2]
        )

        self.assertTrue(res["success"])
        self.assertTrue(res["verification_passed"])

    def test_flower_cvfl_strategy(self):
        """
        Verify CVFLStrategy adapter instantiation and basic parameter aggregation.
        """
        strategy = CVFLStrategy(chunk_size=10, spot_check_ratio=0.1, temperature=5.0)
        self.assertEqual(strategy.chunk_size, 10)
        self.assertEqual(strategy.spot_check_ratio, 0.1)

if __name__ == "__main__":
    unittest.main()

