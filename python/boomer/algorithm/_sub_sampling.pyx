# cython: boundscheck=False
# cython: wraparound=False
from cython.view cimport array as cvarray
from libc.math cimport log2
from sklearn.utils._random import sample_without_replacement
cimport numpy as npc

import numpy as np
from boomer.algorithm.model import DTYPE_INTP
from sklearn.utils import check_random_state


cdef class InstanceSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    cdef uint32[::1] sub_sample(self, float32[::1, :] x, Loss loss, int random_state):
        """
        Creates a sub-sample of the available training examples.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the training examples
        :param loss:            A loss function that should be updated based on the examples included in the sub-sample
        :param random_state:    The seed to be used by RNGs
        :return:                An array of dtype uint, shape `(num_examples)`, representing the weights of the given
                                training examples, i.e., how many times each of the examples is contained in the sample
        """
        pass


cdef class Bagging(InstanceSubSampling):
    """
    Implements bootstrap aggregating (bagging) for drawing a subset (of predefined size) from the available training
    examples with replacement.
    """

    def __cinit__(self, float sample_size = 1):
        """
        :param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available examples)
        """
        self.sample_size = sample_size

    cdef uint32[::1] sub_sample(self, float32[::1, :] x, Loss loss, int random_state):
        cdef intp num_examples = x.shape[0]
        cdef float sample_size = self.sample_size
        cdef int num_samples = <int>(sample_size * num_examples)
        cdef uint32[::1] weights = cvarray(shape=(num_examples,), itemsize=sizeof(uint32), format='I', mode='c')
        weights[:] = 0
        rng = check_random_state(random_state)
        rng_randint = rng.randint
        cdef intp n, i

        # Tell the given loss function that instance sub-sampling is used...
        loss.begin_instance_sub_sampling()

        for n in range(num_samples):
            # Select the index of an example randomly...
             i = rng_randint(num_examples)

             # Update weight at the selected index...
             weights[i] += 1

             # Tell the given loss function that a new example has been chosen to be included in the sample...
             loss.update_sub_sample(i)

        return weights


cdef class FeatureSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling features.
    """

    cdef intp[::1] sub_sample(self, float32[::1, :] x, int random_state):
        """
        Creates a sub-sample of the available features.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the training examples
        :param random_state:    The seed to be used by RNGs
        :return:                An array of dtype int, shape `(num_samples)`, representing the indices of the features
                                contained in the sub-sample
        """
        pass

cdef class RandomFeatureSubsetSelection(FeatureSubSampling):
    """
    Implements random feature subset selection for selecting a random subset (of predefined size) from the available
    features.
    """

    def __cinit__(self, float sample_size = 0):
        """
        :param sample_size: The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available features) or 0, if the default sample size
                            floor(log2(num_features - 1) + 1) should be used
        """
        self.sample_size = sample_size

    cdef intp[::1] sub_sample(self, float32[::1, :] x, int random_state):
         cdef intp num_features = x.shape[1]
         cdef float sample_size = self.sample_size
         cdef int num_samples

         if sample_size > 0:
            num_samples = <int>(sample_size * num_features)
         else:
            num_samples = <int>(log2(num_features - 1) + 1)

         return np.ascontiguousarray(sample_without_replacement(num_features, num_samples, 'auto', random_state),
                                     dtype=DTYPE_INTP)
