#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for implementing multi-label classifiers or rankers.
"""
from skmultilearn.base import MLClassifierBase


class Randomized:
    """
    A base class for all classes that use RNGs.

    Attributes
        random_state   The seed to be used by RNGs
    """

    random_state: int = 0


class Module(Randomized):
    """
    A base class for all modules, a multi-label classifier or ranker consists of.
    """


class MLLearner(MLClassifierBase, Randomized):
    """
    A base class for all multi-label label classifiers or rankers.

    Attributes
        fold    The current fold or None, if no cross validation is used
    """

    fold: int = None

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass
