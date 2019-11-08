# cython: boundscheck=False
# cython: wraparound=False

cdef class InstanceSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    cdef uint8[::1] sub_sample(self, float32[::1, :] x):
        """
        Creates a sub-sample of the available training examples.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :return:    An array of dtype uint, shape `(num_examples)`, representing how many times each of the given
                    training examples is contained in the sub-sample
        """
        pass


cdef class FeatureSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling features.
    """

    cdef intp[::1] sub_sample(self, float32[::1, :] x):
        """
        Creates a sub-sample of the available features.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :return:    An array of dtype int, shape `(num_samples)`, representing the indices of the features contained in
                    the sub-sample
        """
        pass
