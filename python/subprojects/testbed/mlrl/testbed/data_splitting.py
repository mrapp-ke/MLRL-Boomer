"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating machine learning models using either cross validation or separate training
and test sets.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Generator, List

from scipy.sparse import vstack
from sklearn.model_selection import KFold, train_test_split

from mlrl.testbed.experiments.dataset import Dataset, DatasetType
from mlrl.testbed.experiments.fold import Fold, FoldingStrategy
from mlrl.testbed.experiments.input.dataset import DatasetReader
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState


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


class DatasetSplitter(ABC):
    """
    An abstract base class for all classes that split a data set into training and test data.
    """

    @abstractmethod
    def split(self, problem_type: ProblemType) -> Generator[Split]:
        """
        Returns a generator that generates the individual splits of the dataset into training and test data.

        :param problem_type:    The type of the machine learning problem, the dataset is concerned with
        :return:                The generator
        """


class NoSplitter(DatasetSplitter):
    """
    Does not split a dataset.
    """

    class Split(Split):
        """
        A split that does not use separate training and test datasets.
        """

        def __init__(self, state: ExperimentState):
            """
            :param state: The state that stores the dataset
            """
            self.state = state

        def get_state(self, _: DatasetType) -> ExperimentState:
            return self.state

    def __init__(self, dataset_reader: DatasetReader):
        """
        :param dataset_reader: The reader that should be used for loading datasets
        """
        self.dataset_reader = dataset_reader
        self.folding_strategy = FoldingStrategy(num_folds=1, first=0, last=1)
        context = dataset_reader.input_data.context
        context.include_dataset_type = False
        context.include_fold = False

    def split(self, problem_type: ProblemType) -> Generator[Split]:
        log.warning('Not using separate training and test sets. The model will be evaluated on the training data...')
        folding_strategy = self.folding_strategy
        state = ExperimentState(problem_type=problem_type, folding_strategy=folding_strategy)
        self.dataset_reader.read(state)
        state = replace(state, dataset=state.dataset, dataset_type=DatasetType.TRAINING)

        for fold in folding_strategy.folds:
            yield NoSplitter.Split(state=replace(state, fold=fold))


class TrainTestSplitter(DatasetSplitter):
    """
    Splits a dataset into distinct training and test datasets.
    """

    class PredefinedSplit(Split):
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

        def get_state(self, dataset_type: DatasetType) -> ExperimentState:
            state = replace(self.state, dataset_type=dataset_type)
            self.dataset_reader.read(state)
            return state

    class DynamicSplit(Split):
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
            training_dataset: Dataset
            test_dataset: Dataset

        def __init__(self, splitter: 'TrainTestSplitter', state: ExperimentState):
            """
            :param splitter:    The `TrainTestSplitter` that has generated this split
            :param state:       The state that should be used to store the datasets
            """
            self.splitter = splitter
            self.state = state
            context = self.splitter.dataset_reader.input_data.context
            context.include_dataset_type = False

        def get_state(self, dataset_type: DatasetType) -> ExperimentState:
            state = self.state
            splitter = self.splitter
            cache = splitter.cache

            if not cache:
                splitter.dataset_reader.read(state)
                dataset = state.dataset
                x_training, x_test, y_training, y_test = train_test_split(dataset.x,
                                                                          dataset.y,
                                                                          test_size=splitter.test_size,
                                                                          random_state=splitter.random_state,
                                                                          shuffle=True)
                training_dataset = replace(dataset, x=x_training, y=y_training, type=DatasetType.TRAINING)
                test_dataset = replace(dataset, x=x_test, y=y_test, type=DatasetType.TEST)
                cache = TrainTestSplitter.DynamicSplit.Cache(training_dataset=training_dataset,
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
        self.dataset_reader = dataset_reader
        self.test_size = test_size
        self.random_state = random_state
        self.folding_strategy = FoldingStrategy(num_folds=1, first=0, last=1)
        self.cache = None
        context = dataset_reader.input_data.context
        context.include_fold = False
        context.include_dataset_type = True

    def split(self, problem_type: ProblemType) -> Generator[Split]:
        log.info('Using separate training and test sets...')
        dataset_reader = self.dataset_reader
        folding_strategy = self.folding_strategy
        state = ExperimentState(problem_type=problem_type, folding_strategy=folding_strategy)

        # Check if predefined training and test datasets are available...
        predefined_datasets_available = all(
            dataset_reader.is_available(replace(state, dataset_type=dataset_type))
            for dataset_type in [DatasetType.TRAINING, DatasetType.TEST])

        for fold in folding_strategy.folds:
            state = replace(state, fold=fold)

            if predefined_datasets_available:
                yield TrainTestSplitter.PredefinedSplit(dataset_reader=dataset_reader, state=state)
            else:
                yield TrainTestSplitter.DynamicSplit(splitter=self, state=state)


class CrossValidationSplitter(DatasetSplitter):
    """
    Splits the available data into training and test sets corresponding to the individual folds of a cross validation.
    """

    class PredefinedSplit(Split):
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
                        state = replace(state, fold=Fold(index=fold_index))
                        splitter.dataset_reader.read(state)
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
                splitter.dataset_reader.read(state)
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
            state = replace(self.state, dataset_type=dataset_type)
            splitter = self.splitter

            if not splitter.cache:
                splitter.cache = CrossValidationSplitter.PredefinedSplit.Cache(state.folding_strategy.num_folds)

            if dataset_type == DatasetType.TEST:
                dataset = self.__get_test_dataset(state)
            else:
                dataset = self.__get_training_dataset(state)

            return replace(state, dataset=dataset)

    class DynamicSplit(Split):
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
            state = replace(self.state, dataset_type=dataset_type)
            splitter = self.splitter
            dataset_reader = splitter.dataset_reader
            cache = splitter.cache

            if not cache:
                dataset_reader.read(state)
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

    def split(self, problem_type: ProblemType) -> Generator[Split]:
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
