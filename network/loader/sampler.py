from typing import Sequence
import numpy as np

class Sampler:
    def __init__(self, data):
        pass

    def __iter__(self):
        pass

class SubsetRandSampler(Sampler):
    def __init__(self, idx):
        self.idx = idx
    
    def __iter__(self):
        for i in np.random.permutation(len(self.idx)):
            yield self.idx[i]

    def __len__(self):
        return len(self.idx)