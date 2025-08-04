"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for splitting dataset into distinct training and test datasets.
"""
import logging as log

from dataclasses import dataclass, replace
from typing import Any, Generator, Optional, override

from sklearn.model_selection import train_test_split

from mlrl.testbed_sklearn.experiments.dataset import TabularDataset

from mlrl.testbed.experiments.dataset_type import DatasetType
from mlrl.testbed.experiments.fold import FoldingStrategy
from mlrl.testbed.experiments.input.dataset import DatasetReader
from mlrl.testbed.experiments.input.dataset.splitters.splitter import DatasetSplitter
from mlrl.testbed.experiments.state import ExperimentState


class BipartitionSplitter(DatasetSplitter):
    """
    Splits a tabular dataset into distinct training and test datasets.
    """

    class PredefinedSplit(DatasetSplitter.Split):
        """
        A predefined split into a training and a test dataset.
        """

        def __init__(self, dataset_reader: DatasetReader, state: ExperimentState):
            """
            :param dataset_reader:  The reader that should be used for loading datasets
            :param state:           The state that should be used to store the datasets
            """
            self.dataset_reader = dataset_reader
            self.state = state
            context = dataset_reader.input_data.context
            context.include_dataset_type = True

        @override
        def get_state(self, dataset_type: DatasetType) -> ExperimentState:
            """
            See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.Split.get_state`
            """
            return self.dataset_reader.read(replace(self.state, dataset_type=dataset_type))

    class DynamicSplit(DatasetSplitter.Split):
        """
        A split into a training and a test dataset that has been created dynamically.
        """

        @dataclass
        class Cache:
            """
            Caches training and test datasets that been created dynamically.

            Attributes:
                training_dataset:   The training dataset
                test_dataset:       The test dataset
            """
            training_dataset: TabularDataset
            test_dataset: TabularDataset

        def __init__(self, splitter: 'BipartitionSplitter', state: ExperimentState):
            """
            :param splitter:    The `BipartitionSplitter` that has generated this split
            :param state:       The state that should be used to store the datasets
            """
            self.splitter = splitter
            self.state = state
            context = self.splitter.dataset_reader.input_data.context
            context.include_dataset_type = False

        @override
        def get_state(self, dataset_type: DatasetType) -> ExperimentState:
            """
            See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.Split.get_state`
            """
            state = self.state
            splitter = self.splitter
            cache = splitter.cache

            if not cache:
                state = splitter.dataset_reader.read(state)
                dataset = state.dataset
                x_training, x_test, y_training, y_test = train_test_split(dataset.x,
                                                                          dataset.y,
                                                                          test_size=splitter.test_size,
                                                                          random_state=splitter.random_state,
                                                                          shuffle=True)
                training_dataset = replace(dataset, x=x_training, y=y_training)
                test_dataset = replace(dataset, x=x_test, y=y_test)
                cache = BipartitionSplitter.DynamicSplit.Cache(training_dataset=training_dataset,
                                                               test_dataset=test_dataset)
                splitter.cache = cache

            dataset = cache.test_dataset if dataset_type == DatasetType.TEST else cache.training_dataset
            return replace(state, dataset_type=dataset_type, dataset=dataset)

    def __init__(self, dataset_reader: DatasetReader, test_size: float, random_state: int):
        """
        :param dataset_reader:  The reader that should be used for loading datasets
        :param test_size:       The fraction of the available data to be used as the test set
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        """
        super().__init__(FoldingStrategy(num_folds=1, first=0, last=1))
        self.dataset_reader = dataset_reader
        self.test_size = test_size
        self.random_state = random_state
        self.cache: Optional[Any] = None
        context = dataset_reader.input_data.context
        context.include_fold = False
        context.include_dataset_type = True

    @override
    def split(self, state: ExperimentState) -> Generator[DatasetSplitter.Split, None, None]:
        """
        See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.split`
        """
        log.info('Using separate training and test sets...')
        dataset_reader = self.dataset_reader
        folding_strategy = self.folding_strategy
        state = replace(state, folding_strategy=folding_strategy)

        # Check if predefined training and test datasets are available...
        predefined_datasets_available = all(
            dataset_reader.is_available(replace(state, dataset_type=dataset_type))
            for dataset_type in [DatasetType.TRAINING, DatasetType.TEST])

        for fold in folding_strategy.folds:
            state = replace(state, fold=fold)

            if predefined_datasets_available:
                yield BipartitionSplitter.PredefinedSplit(dataset_reader=dataset_reader, state=state)
            else:
                yield BipartitionSplitter.DynamicSplit(splitter=self, state=state)
