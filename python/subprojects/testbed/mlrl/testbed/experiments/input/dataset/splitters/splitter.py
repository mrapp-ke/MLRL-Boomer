"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing splitters that split datasets into training and test datasets.
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional

from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.fold import FoldingStrategy
from mlrl.testbed.experiments.state import ExperimentState


class DatasetSplitter(ABC):
    """
    An abstract base class for all classes that split a dataset into training and test data.
    """

    class Split(ABC):
        """
        An abstract base class for all classes that represent a split of a dataset into training and test datasets.
        """

        @abstractmethod
        def get_state(self, dataset_type: DatasetType) -> Optional[ExperimentState]:
            """
            Returns a state that stores the dataset that corresponds to a specific `DatasetType`.

            :param dataset_type:    The `DatasetType`
            :return:                A state that stores the dataset that corresponds to the given `DatasetType` or None,
                                    if not such dataset is available
            """

    def __init__(self, folding_strategy: FoldingStrategy):
        """
        :param folding_strategy:
        """
        self.folding_strategy = folding_strategy

    @abstractmethod
    def split(self, state: ExperimentState) -> Generator[Split, None, None]:
        """
        Returns a generator that generates the individual splits of the dataset into training and test data.

        :param state: The current state of the experiment
        :return:      The generator
        """
