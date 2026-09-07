from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np

try:
    import flwr as fl
    from flwr.common import (
        Parameters,
        Scalar,
        FitRes,
        NDArrays,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    class FedAvg:
        def __init__(self, **kwargs):
            pass
    Parameters = object
    Scalar = object
    FitRes = object
    ClientProxy = object
    def ndarrays_to_parameters(ndarrays):
        return ndarrays
    def parameters_to_ndarrays(parameters):
        return parameters

from cvfl.core.quantization import SuperIncreasingQuantizer
from cvfl.core.privacy import DualLayerMasking
from cvfl.core.aggregation import SignScoreAggregator
from cvfl.core.commitment import CommitmentManager

class CVFLStrategy(FedAvg):
    """
    Flower Strategy Adapter for CV-FL (Compressed-Verifiable Federated Learning).
    Integrates Hybrid Compression, Dual-Layer Masking, SignScore, and Verification.
    """


    def __init__(
        self,
        chunk_size: int = 10,
        spot_check_ratio: float = 0.1,
        temperature: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.spot_check_ratio = spot_check_ratio
        self.quantizer = SuperIncreasingQuantizer(chunk_size=chunk_size)
        self.privacy = DualLayerMasking()
        self.aggregator = SignScoreAggregator(temperature=temperature)
        self.commitment_mgr = CommitmentManager()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Executes CV-FL aggregation protocol inside Flower framework.
        """
        if not results:
            return None, {}

        # Unpack full parameter vectors from each client
        client_params = [parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results]
        
        # Derive sign vectors for robust weight calculation (sign of each weight)
        client_signs = [np.sign(p).astype(np.int8) for p in client_params]
        
        # Compute robust SignScore weights
        median_sign = self.aggregator.compute_median_sign(client_signs)
        sign_scores = self.aggregator.compute_sign_scores(client_signs, median_sign)
        weights = self.aggregator.compute_robust_weights(sign_scores, adaptive_filter=True)
        
        # Weighted aggregation of the full‑dimensional parameter arrays
        stacked_params = np.stack(client_params, axis=0)  # shape: (N, d)
        # tensordot performs Σ_i weight_i * param_i across the first axis
        weighted_sum = np.tensordot(weights, stacked_params, axes=([0], [0]))
        aggregated_ndarrays = [weighted_sum.astype(np.float32)]
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {
            "round": server_round,
            "mean_sign_score": float(np.mean(sign_scores)),
            "num_clients": len(results),
        }

        return parameters_aggregated, metrics_aggregated
