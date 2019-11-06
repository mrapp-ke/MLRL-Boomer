#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different strategies for sub-sampling training examples and features, such as bagging or
random feature subset selection.
"""
import logging as log
import math
from abc import abstractmethod

import numpy as np
from boomer.algorithm._model import DTYPE_INDICES
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement

from boomer.learners import Randomized


class SubSampling(Randomized):
    """
    A base class for all classes that implement a strategy for sub-sampling training examples or features.
    """

    def _get_sample_indices(self, num_total: int, num_samples: int, with_replacement: bool) -> np.ndarray:
        """
        Returns the indices of the examples or features to be included in the sample.

        :param num_total:           The total number of examples or features
        :param num_samples:         The number of examples or features to be sampled
        :param with_replacement:    True, if the indices should be drawn with replacement, False otherwise
        :return:                    An array of dtype int, shape `(num_samples)`, representing the indices of the
                                    examples or features to be included in the sample
        """
        rng = check_random_state(self.random_state)

        if with_replacement:
            return np.ascontiguousarray(rng.randint(0, num_total, num_samples), dtype=DTYPE_INDICES)
        else:
            return np.ascontiguousarray(sample_without_replacement(num_total, num_samples, random_state=rng),
                                        dtype=DTYPE_INDICES)

    @abstractmethod
    def sub_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Creates and returns sub-sample of the given training data by either sub-sampling the examples or the features.

        :param x:   An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                    training examples
        :return:    An array of dtype int, shape `(num_samples)`, representing the indices of the examples, or the
                    indices of the features, that are included in the sample
        """
        pass


class FeatureSubSampling(SubSampling):
    """
    A base class for all classes that implement a strategy for sub-sampling features.
    """

    def sub_sample(self, x: np.ndarray) -> np.ndarray:
        pass


class RandomFeatureSubSampling(FeatureSubSampling):
    """
    Implement random feature subset selection for selecting a random subset (of predefined size) from the available
    features.
    """

    def __init__(self, sample_size: float = None):
        """
        :param sample_size: The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the features) or None, if the default sample size int(log2(num_features - 1 ) + 1)
                            should be used
        """
        self.sample_size = sample_size

    def __validate(self):
        """
        Raises exceptions if the sub-sampling is not configured properly.
        """
        if self.sample_size is not None:
            if self.sample_size <= 0:
                raise ValueError(
                    'Parameter \'sample_size\' must be None or greater than 0, got {0}'.format(self.sample_size))
            if self.sample_size > 1:
                raise ValueError(
                    'Parameter \'sample_size\' must be None or at maximum 1, got {0}'.format(self.sample_size))

    def sub_sample(self, x: np.ndarray) -> np.ndarray:
        self.__validate()

        # Calculate number of features to be included in the sample
        num_features = x.shape[1]
        sample_size = self.sample_size

        if sample_size is None:
            num_samples = int(math.log(num_features - 1, base=2) + 1)
        else:
            num_samples = int(sample_size * num_features)

        log.debug('Randomly selecting %s out of %s features...', num_samples, num_features)

        # Get indices of features to be included in the sample
        return self._get_sample_indices(num_total=num_features, num_samples=num_samples, with_replacement=False)


class InstanceSubSampling(SubSampling):
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    def sub_sample(self, x: np.ndarray) -> np.ndarray:
        pass


class Bagging(InstanceSubSampling):
    """
    Implements bootstrap aggregating (bagging) for drawing a subset (of predefined size) from the available training
    examples with or without replacement.
    """

    def __init__(self, sample_size: float = 1.0, with_replacement: bool = True):
        """
        :param sample_size:         The fraction of examples to be included in the sample (e.g. a value of 0.6
                                    corresponds to 60 % of the examples)
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

    def sub_sample(self, x: np.ndarray) -> np.ndarray:
        self.__validate()

        # Calculate number of examples to be included in the sample
        num_examples = x.shape[0]
        num_samples = int(self.sample_size * num_examples)
        with_replacement = self.with_replacement

        log.debug('Drawing %s out of %s examples %s replacement...', num_samples, num_examples,
                  'with' if with_replacement else 'without')

        # Get indices of examples to be included in the sample
        return self._get_sample_indices(num_total=num_examples, num_samples=num_samples,
                                        with_replacement=with_replacement)
