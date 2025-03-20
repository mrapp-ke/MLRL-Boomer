"""
This module provides rule learners that make use of the Separate-and-Conquer (SeCo) paradigm.
"""
# pylint: disable=import-self
from mlrl.seco.learners import SeCoClassifier

try:
    from mlrl.seco.testbed.runnables import SeCoRunnable as Runnable
except ImportError:
    # Dependency 'mlrl-testbed' is not available, but that's okay since it's optional.
    pass

SeCo = SeCoClassifier
