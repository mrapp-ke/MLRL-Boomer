"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating machine learning models using either cross validation or separate training
and test sets.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Fold:
    """
    Represents one or several folds of a dataset, i.e., a specific subset of the examples that are contained in the
    dataset.

    Attributes:
        index:          The index of the fold, starting at 0, or None, if no cross validation is used
        num_folds:      The total number of folds
        is_last_fold:   True, if this fold is the last fold, False otherwise
    """
    index: Optional[int]
    num_folds: bool
    is_last_fold: bool

    @property
    def is_cross_validation_used(self) -> bool:
        """
        True, if a cross validation is used, False otherwise.
        """
        return self.num_folds > 1
