"""
This module provides rule learners that make using of the gradient boosting framework.
"""
# pylint: disable=import-self
from mlrl.boosting.boosting_learners import BoomerClassifier

try:
    from mlrl.boosting.runnables import BoomerRunnable as Runnable
except ImportError:
    # Dependency 'mlrl-testbed' is not available, but that's okay since it's optional.
    pass

Boomer = BoomerClassifier
