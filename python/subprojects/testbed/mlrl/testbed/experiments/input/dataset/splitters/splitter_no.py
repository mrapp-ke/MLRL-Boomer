"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that preserve a dataset instead of splitting it into training and test datasets.
"""
import logging as log

from dataclasses import replace
from typing import Generator, Optional, override

from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.fold import FoldingStrategy
from mlrl.testbed.experiments.input.dataset import DatasetReader
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.state import ExperimentState


class NoSplitter(DatasetSplitter):
    """
    Preserves a dataset instead of splitting it into training and test datasets.
    """

    class Split(DatasetSplitter.Split):
        """
        A split that does not use separate training and test datasets.
        """

        def __init__(self, state: ExperimentState):
            """
            :param state: The state that stores the dataset
            """
            self.state = state

        @override
        def get_state(self, dataset_type: DatasetType) -> Optional[ExperimentState]:
            """
            See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.Split.get_state`
            """
            return self.state if dataset_type == DatasetType.TRAINING else None

    def __init__(self, dataset_reader: DatasetReader):
        """
        :param dataset_reader: The reader that should be used for loading datasets
        """
        super().__init__(FoldingStrategy(num_folds=1, first=0, last=1))
        self.dataset_reader = dataset_reader
        context = dataset_reader.input_data.context
        context.include_dataset_type = False
        context.include_fold = False

    @override
    def split(self, state: ExperimentState) -> Generator[DatasetSplitter.Split, None, None]:
        """
        See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.split`
        """
        log.warning('Not using separate training and test sets. The model will be evaluated on the training data...')
        folding_strategy = self.folding_strategy
        state = replace(state, folding_strategy=folding_strategy)
        state = self.dataset_reader.read(state)
        state = replace(state, dataset_type=DatasetType.TRAINING)

        for fold in folding_strategy.folds:
            yield NoSplitter.Split(state=replace(state, fold=fold))
