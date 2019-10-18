#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different strategies for sub-sampling training examples and features, such as bagging or
random feature subset selection.
"""
import logging as log
from abc import abstractmethod

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement

from boomer.algorithm.stats import get_num_examples
from boomer.learners import Randomized


class InstanceSubSampling(Randomized):
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    @abstractmethod
    def sub_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Creates a sub-sample of the given training examples.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :return:    An array of dtype int, shape `(num_samples)`, representing the indices of the examples that are
                    included in the sample
        """
        pass


class Bagging(InstanceSubSampling):

    def __init__(self, sample_size: float = 1.0, with_replacement: bool = True):
        """
        :param sample_size:         The fraction of examples to be included in the sample, e.g. a value of 0.6
                                    corresponds to 60 % of the examples
        :param with_replacement:    'True', if the examples should be drawn with replacement, 'False' otherwise
        """
        self.sample_size = sample_size
        self.with_replacement = with_replacement

    def __validate(self):
        """
        Raises exceptions if the sub-sampling is not configured properly.
        """
        if self.sample_size <= 0:
            raise ValueError('Parameter \'sample_size\' must be greater than 0, got {0}'.format(self.sample_size))
        if self.sample_size > 1:
            raise ValueError('Parameter \'sample_size\' must be at maximum 1, got {0}'.format(self.sample_size))

    def __get_sample_indices(self, total_examples: int, num_samples: int) -> np.ndarray:
        """
        Returns the indices of the examples to be included in the sample.

        :param total_examples:  The total number of examples
        :param num_samples:     The number of examples to be sampled
        :return:                An array of dtype int, shape `(num_samples)`, representing the indices of the examples
                                to be included in the sample
        """
        rng = check_random_state(self.random_state)

        if self.with_replacement:
            indices = rng.randint(0, total_examples, num_samples)
        else:
            indices = sample_without_replacement(total_examples, num_samples, random_state=rng)

        return indices

    def sub_sample(self, x: np.ndarray) -> np.ndarray:
        self.__validate()

        # Calculate number of examples to be included in the sample
        num_examples = get_num_examples(x)
        num_samples = int(self.sample_size * num_examples)

        log.debug('Drawing %s out of %s examples %s replacement...', num_samples, num_examples,
                  'with' if self.with_replacement else 'without')

        # Get indices of examples that are included in the sample
        return self.__get_sample_indices(num_examples, num_samples)
