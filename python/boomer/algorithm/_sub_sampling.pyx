# cython: boundscheck=False
# cython: wraparound=False
from cython.view cimport array as cvarray
from libc.math cimport log2


cdef class InstanceSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    cdef uint8[::1] sub_sample(self, float32[::1, :] x, int random_state):
        """
        Creates a sub-sample of the available training examples.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the training examples
        :param random_state:    The seed to be used by RNGs
        :return:                An array of dtype uint, shape `(num_examples)`, representing the weights of the given
                                training examples, i.e., how many times each of the examples is contained in the sample
        """
        pass


cdef class Bagging(InstanceSubSampling):
    """
    Implements bootstrap aggregating (bagging) for drawing a subset (of predefined size) from the available training
    examples with or without replacement.
    """

    def __cinit__(self, float sample_size = 1, bint with_replacement = 1):
        """
        :param sample_size:         The fraction of examples to be included in the sample (e.g. a value of 0.6
                                    corresponds to 60 % of the available examples)
        :param with_replacement:    1, if the examples should be drawn with replacement, 0 otherwise
        """
        self.sample_size = sample_size
        self.with_replacement = with_replacement

    cdef uint8[::1] sub_sample(self, float32[::1, :] x, int random_state):
        cdef intp num_examples = x.shape[0]
        cdef float sample_size = self.sample_size
        cdef int num_samples = <int>(sample_size * num_examples)
        cdef uint8[::1] weights = cvarray(shape=(num_samples,), itemsize=sizeof(uint8), format='H', mode='c')
        cdef bint with_replacement = self.with_replacement
        # TODO Random part
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
                            floor(log2(num_features) - 1) + 1 should be used
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

         cdef intp[::1] feature_indices = cvarray(shape=(num_samples,), itemsize=sizeof(intp), format='n', mode='c')
         # TODO random part
         return feature_indices
