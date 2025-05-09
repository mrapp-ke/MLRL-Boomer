"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing splitters that split datasets into training and test datasets.
"""

from abc import ABC, abstractmethod
from typing import Generator

from mlrl.testbed.experiments.dataset import DatasetType
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState


class DatasetSplitter(ABC):
    """
    An abstract base class for all classes that split a data set into training and test data.
    """

    class Split(ABC):
        """
        An abstract base class for all classes that represent a split of a dataset into training and test datasets.
        """

        @abstractmethod
        def get_state(self, dataset_type: DatasetType) -> ExperimentState:
            """
            Returns a state that stores the dataset that corresponds to a specific `DatasetType`.

            :param dataset_type:    The `DatasetType`
            :return:                A state that stores the dataset that corresponds to the given `DatasetType`
            """

    @abstractmethod
    def split(self, problem_type: ProblemType) -> Generator[Split, None, None]:
        """
        Returns a generator that generates the individual splits of the dataset into training and test data.

        :param problem_type:    The type of the machine learning problem, the dataset is concerned with
        :return:                The generator
        """
