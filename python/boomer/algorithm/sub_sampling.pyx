"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for sub-sampling training examples or features.
"""
from boomer.algorithm._arrays cimport array_uint32

from libc.math cimport log2

import numpy as np
from sklearn.utils._random import sample_without_replacement
from boomer.algorithm.model import DTYPE_INTP
from sklearn.utils import check_random_state


cdef class InstanceSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    cdef uint32[::1] sub_sample(self, intp num_examples, RNG rng):
        """
        Creates and returns a sub-sample of the available training examples.

        :param x:   The total number of available training examples
        :param rng: The random number generator to be used
        :return:    An array of dtype uint, shape `(num_examples)`, representing the weights of the given training
                    examples, i.e., how many times each of the examples is contained in the sample
        """
        pass


cdef class Bagging(InstanceSubSampling):
    """
    Implements bootstrap aggregating (bagging) for drawing a subset (of predefined size) from the available training
    examples with replacement.
    """

    def __cinit__(self, float32 sample_size = 1.0):
        """
        :param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available examples)
        """
        self.sample_size = sample_size

    cdef uint32[::1] sub_sample(self, intp num_examples, RNG rng):
        cdef float32 sample_size = self.sample_size
        cdef intp num_samples = <intp>(sample_size * num_examples)
        cdef uint32[::1] weights = array_uint32(num_examples)
        cdef uint32 random_index
        cdef intp i

        weights[:] = 0

        for i in range(num_samples):
            # Randomly select the index of an example...
            random_index = rng.random(0, num_examples)

            # Update weight at the selected index...
            weights[random_index] += 1

        return weights


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):
    """
    Implements random instance subset selection for drawing a subset (of predefined size) from the available training
    examples without replacement.
    """

    def __cinit__(self, float32 sample_size = 0.66):
        """
        param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                           60 % of the available examples)
        """
        self.sample_size = sample_size

    cdef uint32[::1] sub_sample(self, intp num_examples, RNG rng):
        cdef float32 sample_size = self.sample_size
        cdef intp num_samples = <intp>(sample_size * num_examples)
        cdef uint32 limit = num_examples
        cdef uint32[::1] weights = array_uint32(num_examples)
        cdef uint32[::1] indices = array_uint32(num_examples)
        cdef uint32 random_index
        cdef intp i

        # Initialize arrays...
        for i in range(num_examples):
            weights[i] = 0
            indices[i] = i

        for i in range(num_examples):
            # Randomly select an index that has not been drawn yet, i.e., which belongs to the region [0, limit)...
            random_index = indices[rng.random(0, limit)]

            # Set weight at the selected index to 1...
            weights[random_index] = 1

            # Shrink the region [0, limit) that contains the indices of the examples that have not been drawn yet and
            # move the element at the border to the position of the recently drawn element...
            limit -= 1
            indices[random_index] = indices[limit]

        return weights


cdef class FeatureSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling features.
    """

    cdef intp[::1] sub_sample(self, intp num_features, intp random_state):
        """
        Creates and returns a sub-sample of the available features.

        :param num_features:    The total number of available features
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

    def __cinit__(self, float32 sample_size = 0.0):
        """
        :param sample_size: The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available features) or 0, if the default sample size
                            floor(log2(num_features - 1) + 1) should be used
        """
        self.sample_size = sample_size

    cdef intp[::1] sub_sample(self, intp num_features, intp random_state):
         cdef float32 sample_size = self.sample_size
         cdef intp num_samples

         if sample_size > 0:
            num_samples = <intp>(sample_size * num_features)
         else:
            num_samples = <intp>(log2(num_features - 1) + 1)

         return np.ascontiguousarray(sample_without_replacement(num_features, num_samples, 'auto', random_state),
                                     dtype=DTYPE_INTP)


cdef class LabelSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling labels.
    """

    cdef intp[::1] sub_sample(self, intp num_labels, intp random_state):
        """
        Creates and returns a sub-sample of the available labels.
        
        :param num_labels:      The total number of available labels
        :param random_state:    The seed to be used by RNGs
        :return:                An array of dtype int, shape `(num_samples)`, representing the indices of the labels 
                                contained in the sub-sample
        """
        pass


cdef class RandomLabelSubsetSelection(LabelSubSampling):

    def __cinit__(self, intp num_samples):
        """
        :param num_samples: The number of labels to be included in the sample
        """
        self.num_samples = num_samples

    cdef intp[::1] sub_sample(self, intp num_labels, intp random_state):
        cdef intp num_samples = self.num_samples
        return np.ascontiguousarray(sample_without_replacement(num_labels, num_samples, 'auto', random_state),
                                    dtype=DTYPE_INTP)
