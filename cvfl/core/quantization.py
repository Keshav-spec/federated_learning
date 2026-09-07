import numpy as np
import torch

class SuperIncreasingQuantizer:
    """
    Ternary Sign Quantization and Super-Increasing Sequence Compression.
    Maps continuous gradient updates g_i in R^d -> sign(g_i) in {-1, 0, 1}^d -> trits {0, 1, 2}^d.
    Packs trits into compact scalar representations using super-increasing powers of 3: s_k = 3^(k-1).
    """

    def __init__(self, chunk_size: int = 10):
        """
        :param chunk_size: Number of trits per packed integer (default 10 trits: max value 3^10-1 = 59048).
        """
        self.chunk_size = chunk_size
        self.powers_of_3 = np.array([3 ** i for i in range(chunk_size)], dtype=np.int64)

    def quantize_to_signs(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Quantize continuous tensor/gradients into ternary sign representation {-1, 0, 1}.
        """
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.array(tensor)
        arr_clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        signs = np.sign(arr_clean).astype(np.int8)
        return signs

    def signs_to_trits(self, signs: np.ndarray) -> np.ndarray:
        """
        Remap ternary signs {-1, 0, 1} to non-negative base-3 trits {0, 1, 2}.
        """
        return (signs + 1).astype(np.uint8)

    def trits_to_signs(self, trits: np.ndarray) -> np.ndarray:
        """
        Remap base-3 trits {0, 1, 2} back to ternary signs {-1, 0, 1}.
        """
        return (trits.astype(np.int8) - 1)

    def pack(self, trits: np.ndarray) -> np.ndarray:
        """
        Compress trit array using super-increasing sequence s_k = 3^k.
        :param trits: 1D array of uint8 trits {0, 1, 2}.
        :return: 1D array of packed integer scalars.
        """
        d = len(trits)
        num_chunks = int(np.ceil(d / self.chunk_size))
        pad_len = num_chunks * self.chunk_size - d
        if pad_len > 0:
            padded_trits = np.pad(trits, (0, pad_len), mode='constant', constant_values=1) # 1 represents sign 0
        else:
            padded_trits = trits
        
        reshaped = padded_trits.reshape(num_chunks, self.chunk_size)
        packed = np.dot(reshaped, self.powers_of_3)
        return packed

    def unpack(self, packed: np.ndarray, original_d: int) -> np.ndarray:
        """
        Reversibly decompress packed scalars back into base-3 trits.
        :param packed: 1D array of packed integer scalars.
        :param original_d: Target length of original trit vector.
        :return: 1D array of uint8 trits.
        """
        num_chunks = len(packed)
        trits_matrix = np.zeros((num_chunks, self.chunk_size), dtype=np.uint8)
        
        vals = packed.copy()
        for i in range(self.chunk_size):
            trits_matrix[:, i] = (vals % 3).astype(np.uint8)
            vals = vals // 3
            
        unpacked_flat = trits_matrix.reshape(-1)
        return unpacked_flat[:original_d]

    def compress_gradient(self, tensor: torch.Tensor):
        """
        Full pipeline: Tensor -> Sign -> Trits -> Packed Scalars.
        """
        signs = self.quantize_to_signs(tensor)
        trits = self.signs_to_trits(signs)
        packed = self.pack(trits)
        return signs, trits, packed

    def decompress_gradient(self, packed: np.ndarray, original_d: int) -> np.ndarray:
        """
        Full reverse pipeline: Packed Scalars -> Trits -> Signs.
        """
        trits = self.unpack(packed, original_d)
        signs = self.trits_to_signs(trits)
        return signs
