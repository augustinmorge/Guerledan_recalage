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

    def __multinomial(self, samples, N):
        # Get list with only weights
        weights = samples[0]

        # Compute cumulative sum
        Q = cumulative_sum(weights)
        # Q = np.cumsum(weights)

        # As long as the number of new samples is insufficient
        n = 0
        new_samples = []
        while n < N:

            # Draw a random sample u
            u = np.random.uniform(0, 1, 1)[0]

            # Naive search (alternative: binary search)
            # m = naive_search(Q, u)
            m = 0
            while Q[m] < u:
                m += 1

            # Add copy of the state sample (uniform weights)
            new_samples.append([1.0/N, samples[1][0][m][0], samples[1][1][m][0]])

            # Added another sample
            n += 1


        new_samples = np.array(new_samples)
        new_samples = [new_samples[:,0].reshape(N,1), [new_samples[:,1].reshape(N,1), new_samples[:,2].reshape(N,1)]]

        return new_samples

    def resample(self, samples, N):
        """
        Particles should at least be present floor(wi/N) times due to first deterministic loop. First Nt new samples are
        always the same (when running the function with the same input multiple times).

        Computational complexity: O(M) + O(N-Nt), where Nt is number of samples in first deterministic loop

        :param samples: Samples that must be resampled.
        :param N: Number of samples that must be generated.
        :return: Resampled weighted particles.
        """
        # Copy sample and ease of writing
        wm, xm = copy.deepcopy(samples)

        # Compute replication
        Nm = np.floor(N * wm)

        # Store weight adjusted sample (and avoid division of integers)
        weight_adjusted_samples = [wm - Nm.astype(float)/N, xm]

        # Store sample to be used for replication
        Nm = Nm.astype(int)
        replication_samples = [xm, Nm]

        # Replicate samples
        new_samples_deterministic = []
        Nt = np.sum(Nm)
        val_x = np.zeros((Nt,1))
        val_y = np.zeros((Nt,1))
        import time
        for i in range(Nm.shape[0]):
            idx_pre = np.sum(Nm[:i,])
            idx_now = np.sum(Nm[:i+1,])
            if idx_now - idx_pre != 0:
                val_x[idx_pre:idx_now,] = xm[0][i,0]*np.ones((Nm[i,0], 1))
                val_y[idx_pre:idx_now,] = xm[1][i,0]*np.ones((Nm[i,0], 1))
            # print(f"So, Nm = {Nm}\nwith idx_pre, idx_now = {idx_pre, idx_now}")
            # print(f"val_x={val_x}")
            # time.sleep(0.5)

        new_samples_deterministic.append(val_x)
        new_samples_deterministic.append(val_y)

        # Normalize new weights if needed
        if N != Nt:
            weight_adjusted_samples[0] *= float(N) / (N - Nt)

        # Resample remaining samples (__multinomial return weighted samples, discard weights)
        new_samples_stochastic = self.__multinomial(weight_adjusted_samples, N - Nt)[1]

        # Return new samples
        new_x = np.vstack((new_samples_deterministic[0], new_samples_stochastic[0]))
        new_y = np.vstack((new_samples_deterministic[1], new_samples_stochastic[1]))
        weighted_new_samples = [1.0/N*np.ones((N,1)), [new_x, new_y]]

        # print(new_samples_deterministic)
        return weighted_new_samples
