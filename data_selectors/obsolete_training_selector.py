from collections import defaultdict

import numpy as np

from exceptions import SanityCheckError


class ObsoleteTrainingSelector:
    "This class is pretty much deprecated for now, I will use a different approach.b"
    def __init__(self, inner_radius, outer_radius=0, ignore_downstream=True,):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.ignore_downstream = ignore_downstream
        self.seen_samples = defaultdict(int)
        self.mask = None     

    @property
    def radius(self):
        return self.inner_radius + self.outer_radius

    def get_seen_samples(self):
        repeat_samples = {k: v for k,v in self.seen_samples.items() if v > 1}
        return dict(sorted(repeat_samples.items(), key=lambda x: x[1], reverse=True))

    def get_training_examples(self, ansi_array):
        x = []
        padded_ansi = PaddedArrayWrapper.setup(ansi_array, self.radius)
        for i in range(ansi_array.shape[0]):
            for j in range(ansi_array.shape[1]):
                array_window = padded_ansi.get_slice_from_index(i, j)
                source_example = self.process_array_slice(array_window)
                self.seen_samples[source_example] += 1
                if np.random.uniform() <= pow(self.seen_samples[source_example], -0.66666):
                    x.append(source_example)
        return x

    def process_array_slice(self, array_slice: np.array):
        masked_array_slice = self._mask_array_slice(array_slice)

        flat_slice = [number for number in masked_array_slice.flatten() if number != -1]

        if flat_slice[-1] == 65536:
            raise SanityCheckError("65536 is an invalid centre ansi char")
        #if not self.ignore_downstream:
            # when getting everything
            #return tuple(np.concatenate([
            #    flat_slice[:flat_slice.shape[0] // 2],
            #    flat_slice[flat_slice.shape[0] // 2 + 1:],
            #    flat_slice[flat_slice.shape[0] // 2:flat_slice.shape[0]//2 + 1],
            #]))

        return tuple(flat_slice)


    def _mask_array_slice(self, array_slice: np.array):
        if self.mask is None:
            mask = np.ones(array_slice.shape)
            big_i = mask.shape[0]
            big_j = mask.shape[1]
            for i in range(big_i):
                for j in range(big_j):
                    if self.ignore_downstream and (i > big_i // 2 or i == big_i // 2 and j > big_j // 2):
                        mask[i][j] = 0
                    elif not max(abs(i - big_i // 2), abs(j - big_j // 2)) <= self.inner_radius:
                        mask[i][j] = (i + j) % 2
            self.mask = mask

        array_slice = self.mask * array_slice - (1 - self.mask)
        return array_slice

class PaddedArrayWrapper:
    def __init__(self, padded_array, radius):
        self.padded_array = padded_array
        self.radius = radius

    @classmethod
    def setup(cls, array, radius, padding_value=65536):
        self = cls.__new__(cls)
        padded_array = np.pad(array, radius, constant_values=padding_value)
        self.__init__(padded_array, radius)
        return self

    def get_slice_from_index(self, i, j):
        return self.padded_array[i: i + 2*self.radius + 1, j: j + 2*self.radius + 1]



