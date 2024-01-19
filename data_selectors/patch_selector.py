import numpy as np

class PatchSelector:
    def select_patches_from_int(self, ansi: np.ndarray):
        vectorized_patches = []
        length_of_ansi = ansi.shape[0]
        padded_ansi = np.pad(
            ansi,
            ((0, 1), (0, 0), (0, 0)), constant_values=0
        )
        for i in range(-(-ansi.shape[0] // 2)):
            for j in range(20):
                flat_patch = padded_ansi[i:i+2, j:j+4].flatten()
                vectorized_patches.append(flat_patch)

        return vectorized_patches
