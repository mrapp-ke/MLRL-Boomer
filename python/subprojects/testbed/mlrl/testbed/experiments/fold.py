"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing folds of a dataset.
"""
from dataclasses import dataclass
from typing import Generator


@dataclass
class Fold:
    """
    Represents an individual fold of a dataset, i.e., a specific subset of the examples that are contained in the
    dataset.

    Attributes:
        index: The index of the fold
    """
    index: int


@dataclass
class FoldingStrategy:
    """
    Represents a strategy for folding a dataset, i.e., for creating different subsets of the examples that are contained
    in the dataset.

    Attributes:
        num_folds:  The total number of folds
        first:      The index of the first fold to be created (inclusive)
        last:       The index of the last fold to be created (exclusive)
    """
    num_folds: int
    first: int
    last: int

    @property
    def folds(self) -> Generator[Fold, None, None]:
        """
        Returns a generator that generates all folds that are created by the strategy.

        :return: The generator
        """
        for index in range(self.first, self.last):
            yield Fold(index=index)

    @property
    def num_folds_in_subset(self) -> int:
        """
        The number of folds that are actually created.
        """
        return self.last - self.first

    @property
    def is_subset(self):
        """
        True, if only a subset of the folds is actually created, False otherwise.
        """
        return self.num_folds_in_subset < self.num_folds

    @property
    def is_cross_validation_used(self) -> bool:
        """
        True, if a cross validation is used, False otherwise.
        """
        return self.num_folds > 1

    def is_last_fold(self, fold: 'Fold') -> bool:
        """
        Returns whether a given fold is the last fold.

        :param fold:    The fold to be checked
        :return:        True, if the given fold is the last fold, False otherwise
        """
        return fold.index == self.last - 1
