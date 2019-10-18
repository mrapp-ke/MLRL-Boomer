#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different strategies for sub-sampling training examples and features, such as bagging or
random feature subset selection.
"""
import numpy as np

from boomer.learners import Randomized


class InstanceSubSampling(Randomized):
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    def subsample(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Creates a sub-sample of the given training examples.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :return:    An array of dtype float, shape `(num_samples, num_features)`, representing the sub-sampled examples,
                    as well as an array of dtype float, shape `(num_out_of_sample, num_features)`, representing the
                    examples that have not been sampled, if available
        """
        pass
