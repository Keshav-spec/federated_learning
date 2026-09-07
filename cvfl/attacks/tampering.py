import numpy as np
from typing import List, Tuple

class AggregatorTamperSimulator:
    """
    Simulates a dishonest Edge Aggregator (EA) or Cloud Server (CS) (G3 - Verifiability).
    The dishonest aggregator attempts to falsify aggregated updates or client contributions.
    """

    def __init__(self, tamper_rate: float = 0.1, tamper_mode: str = "sign_flip"):
        """
        :param tamper_rate: Fraction of vector dimensions/chunks tampered.
        :param tamper_mode: 'sign_flip', 'zero_out', or 'random_noise'.
        """
        self.tamper_rate = tamper_rate
        self.tamper_mode = tamper_mode.lower()

    def tamper_aggregate_signs(self, aggregate_signs: np.ndarray) -> np.ndarray:
        """
        Tamper with reported aggregated sign vector.
        """
        if self.tamper_rate <= 0.0:
            return aggregate_signs

        tampered = aggregate_signs.copy()
        d = len(tampered)
        num_tamper = max(1, int(d * self.tamper_rate))
        indices = np.random.choice(d, size=num_tamper, replace=False)

        if self.tamper_mode == "sign_flip":
            tampered[indices] = -tampered[indices]
        elif self.tamper_mode == "zero_out":
            tampered[indices] = 0
        elif self.tamper_mode == "random_noise":
            tampered[indices] = np.random.choice([-1, 0, 1], size=num_tamper)

        return tampered
