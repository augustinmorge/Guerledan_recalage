#!/usr/bin/env python

# Numpy
import numpy as np

# Enum
from enum import Enum

# Deep copy samples
import copy

# Helper functions
from resampling_helpers import *


class ResamplingAlgorithms(Enum):
    MULTINOMIAL = 1
    RESIDUAL = 2
    STRATIFIED = 3
    SYSTEMATIC = 4


class Resampler:
    """
    Resample class that implements different resampling methods.
    """

    def __init__(self):
        self.initialized = True

    def resample(self, samples, N):

        # Get list with only weights
        weights = samples[0]

        # Compute cumulative sum
        Q = cumulative_sum(weights)
        # print(samples)

        # As long as the number of new samples is insufficient
        n = 0
        new_samples = []
        while n < N:

            # Draw a random sample u
            u = np.random.uniform(1e-6, 1, 1)[0]

            # Naive search (alternative: binary search)
            m = naive_search(Q, u)

            # Add copy of the state sample (uniform weights)
            new_samples.append([1.0/N, samples[1][0][m][0], samples[1][1][m][0]])

            # Added another sample
            n += 1

        new_samples = np.array(new_samples)
        new_samples = [new_samples[:,0].reshape(N,1), [new_samples[:,1].reshape(N,1), new_samples[:,2].reshape(N,1)]]

        return new_samples
