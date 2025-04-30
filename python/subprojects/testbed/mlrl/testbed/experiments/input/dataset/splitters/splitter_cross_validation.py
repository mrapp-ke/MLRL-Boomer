"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for splitting datasets into multiple, equally sized, folds consisting of a training and a test dataset.
"""
import logging as log

from dataclasses import dataclass, field, replace
from typing import Generator, List

from scipy.sparse import vstack
from sklearn.model_selection import KFold

from mlrl.testbed.experiments.dataset import Dataset, DatasetType
from mlrl.testbed.experiments.fold import Fold, FoldingStrategy
from mlrl.testbed.experiments.input.dataset import DatasetReader
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState


class CrossValidationSplitter(DatasetSplitter):
    """
    Splits the available data into training and test sets corresponding to the individual folds of a cross validation.
    """

    class PredefinedSplit(DatasetSplitter.Split):
        """
        A predefined split into training and test datasets that corresponds to an individual fold of a cross validation.
        """

        class Cache:
            """
            Caches the datasets that correspond to individual folds of a cross validation.
            """

            def __init__(self, num_folds: int):
                """
                :param num_folds: The total number of folds
                """
                self.datasets = [None for _ in range(num_folds)]

        def __get_training_dataset(self, state: ExperimentState) -> Dataset:
            splitter = self.splitter
            cache = splitter.cache
            training_dataset = None

            for fold_index in range(state.folding_strategy.num_folds):
                if fold_index != state.fold.index:
                    dataset = cache.datasets[fold_index]

                    if not dataset:
                        state = splitter.dataset_reader.read(replace(state, fold=Fold(index=fold_index)))
                        dataset = state.dataset
                        cache.datasets[fold_index] = dataset

                    if training_dataset:
                        training_dataset.x = vstack((training_dataset.x, dataset.x))
                        training_dataset.y = vstack((training_dataset.y, dataset.y))
                    else:
                        training_dataset = replace(dataset, type=DatasetType.TRAINING)

            return training_dataset

        def __get_test_dataset(self, state: ExperimentState) -> Dataset:
            splitter = self.splitter
            cache = splitter.cache
            fold_index = state.fold.index
            dataset = cache.datasets[fold_index]

            if not dataset:
                state = splitter.dataset_reader.read(state)
                dataset = state.dataset
                cache.datasets[fold_index] = dataset

            return replace(dataset, type=DatasetType.TEST)

        def __init__(self, splitter: 'CrossValidationSplitter', state: ExperimentState):
            """
            :param splitter:    The `CrossValidationSplitter` that has created this split
            :param state:       The state that should be used to store the datasets
            """
            self.splitter = splitter
            self.state = state
            context = splitter.dataset_reader.input_data.context
            context.include_dataset_type = False
            context.include_fold = True

        def get_state(self, dataset_type: DatasetType) -> ExperimentState:
            """
            See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.Split.get_state`
            """
            state = replace(self.state, dataset_type=dataset_type)
            splitter = self.splitter

            if not splitter.cache:
                splitter.cache = CrossValidationSplitter.PredefinedSplit.Cache(state.folding_strategy.num_folds)

            if dataset_type == DatasetType.TEST:
                dataset = self.__get_test_dataset(state)
            else:
                dataset = self.__get_training_dataset(state)

            return replace(state, dataset=dataset)

    class DynamicSplit(DatasetSplitter.Split):
        """
        A split into training and test datasets that corresponds to an individual fold of a cross validation and is
        created dynamically.
        """

        @dataclass
        class Cache:
            """
            Caches training and test datasets that correspond to individual folds.

            Attributes:
                 training_datasets: A list that stores the training datasets
                 test_datasets:     A list that stores the test datasets
            """
            training_datasets: List[Dataset] = field(default_factory=list)
            test_datasets: List[Dataset] = field(default_factory=list)

        def __init__(self, splitter: 'CrossValidationSplitter', state: ExperimentState):
            """
            :param splitter:    The `CrossValidationSplitter` that has created this split
            :param state:       The state that should be used to store the datasets
            """
            self.splitter = splitter
            self.state = state
            context = splitter.dataset_reader.input_data.context
            context.include_dataset_type = False
            context.include_fold = False

        def get_state(self, dataset_type: DatasetType) -> ExperimentState:
            """
            See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.Split.get_state`
            """
            state = replace(self.state, dataset_type=dataset_type)
            splitter = self.splitter
            dataset_reader = splitter.dataset_reader
            cache = splitter.cache

            if not cache:
                state = dataset_reader.read(state)
                dataset = state.dataset
                cache = CrossValidationSplitter.DynamicSplit.Cache()
                splits = KFold(n_splits=state.folding_strategy.num_folds,
                               random_state=splitter.random_state,
                               shuffle=True).split(dataset.x, dataset.y)

                for training_examples, test_examples in splits:
                    training_dataset = replace(dataset,
                                               x=dataset.x[training_examples],
                                               y=dataset.y[training_examples],
                                               type=DatasetType.TRAINING)
                    cache.training_datasets.append(training_dataset)
                    test_dataset = replace(dataset,
                                           x=dataset.x[test_examples],
                                           y=dataset.y[test_examples],
                                           type=DatasetType.TEST)
                    cache.test_datasets.append(test_dataset)

                splitter.cache = cache

            datasets = cache.test_datasets if dataset_type == DatasetType.TEST else cache.training_datasets
            return replace(state, dataset=datasets[state.fold.index])

    def __init__(self, dataset_reader: DatasetReader, num_folds: int, current_fold: int, random_state: int):
        """
        :param dataset_reader:  The reader that should be used for loading datasets
        :param num_folds:       The total number of folds to be used by cross validation or 1, if separate training and
                                test sets should be used
        :param current_fold:    The cross validation fold to be performed or -1, if all folds should be performed
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        """
        self.dataset_reader = dataset_reader
        self.random_state = random_state
        self.folding_strategy = FoldingStrategy(num_folds=num_folds,
                                                first=0 if current_fold < 0 else current_fold,
                                                last=num_folds if current_fold < 0 else current_fold + 1)
        self.cache = None
        context = dataset_reader.input_data.context
        context.include_dataset_type = False
        context.include_fold = True

    def split(self, problem_type: ProblemType) -> Generator[DatasetSplitter.Split]:
        """
        See :func:`mlrl.testbed.experiments.input.dataset.splitters.splitter.DatasetSplitter.split`
        """
        folding_strategy = self.folding_strategy
        num_folds = folding_strategy.num_folds

        log.info(
            'Performing %s %s-fold cross validation...', 'fold ' + str(folding_strategy.first + 1) +
            (' to ' + str(folding_strategy.last) if folding_strategy.num_folds_in_subset > 1 else '')
            + ' of' if folding_strategy.is_subset else 'full', num_folds)

        # Check if predefined folds are available...
        state = ExperimentState(problem_type=problem_type, folding_strategy=folding_strategy)
        predefined_splits_available = all(
            self.dataset_reader.is_available(replace(state, fold=Fold(fold_index))) for fold_index in range(num_folds))

        for fold in folding_strategy.folds:
            log.info('Fold %s / %s:', (fold.index + 1), num_folds)
            state = replace(state, fold=fold)

            if predefined_splits_available:
                yield CrossValidationSplitter.PredefinedSplit(splitter=self, state=state)
            else:
                yield CrossValidationSplitter.DynamicSplit(splitter=self, state=state)
