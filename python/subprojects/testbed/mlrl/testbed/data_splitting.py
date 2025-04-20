"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating machine learning models using either cross validation or separate training
and test sets.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generator

from scipy.sparse import vstack
from sklearn.model_selection import KFold, train_test_split

from mlrl.testbed.experiments.dataset import Dataset, DatasetType
from mlrl.testbed.experiments.fold import Fold, FoldingStrategy
from mlrl.testbed.experiments.input.dataset import DatasetReader
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState


@dataclass
class Split:
    """
    A split of a dataset into training and test datasets.

    :param folding_strategy:    The strategy that is used for creating folds
    :param fold:                The fold, the split corresponds to
    :param training_dataset:    The training dataset
    :param test_dataset:        The test dataset
    """
    folding_strategy: FoldingStrategy
    fold: Fold
    training_dataset: Dataset
    test_dataset: Dataset


class DatasetSplitter(ABC):
    """
    An abstract base class for all classes that split a data set into training and test data.
    """

    @abstractmethod
    def split(self) -> Generator[Split]:
        """
        Returns a generator that generates the individual splits of the dataset into training and test data.

        :return: The generator
        """


class NoSplitter(DatasetSplitter):
    """
    Does not split the available data into separate train and test sets.
    """

    def __init__(self, dataset_reader: DatasetReader):
        """
        :param dataset_reader: The reader that should be used for loading datasets
        """
        self.dataset_reader = dataset_reader
        self.folding_strategy = FoldingStrategy(num_folds=1, first=0, last=1)

    def split(self) -> Generator[Split]:
        log.warning('Not using separate training and test sets. The model will be evaluated on the training data...')

        # Load dataset...
        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)
        dataset_reader = self.dataset_reader
        context = dataset_reader.input_data.context
        context.include_dataset_type = False
        context.include_fold = False
        dataset_reader.read(state)
        dataset = state.dataset

        # Train and evaluate model...
        folding_strategy = self.folding_strategy

        for fold in folding_strategy.folds:
            yield Split(folding_strategy=folding_strategy, fold=fold, training_dataset=dataset, test_dataset=dataset)


class TrainTestSplitter(DatasetSplitter):
    """
    Splits the available data into a single train and test set.
    """

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

    def split(self) -> Generator[Split]:
        log.info('Using separate training and test sets...')

        # Check if predefined training and test datasets are available...
        dataset_reader = self.dataset_reader
        context = dataset_reader.input_data.context
        context.include_dataset_type = True
        context.include_fold = False

        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)

        context.include_dataset_type = dataset_reader.is_available(state) and dataset_reader.is_available(
            replace(state, dataset_type=DatasetType.TEST))

        # Load (training) dataset...
        dataset_reader.read(state)
        training_dataset = state.dataset

        if context.include_dataset_type:
            # Load test dataset...
            state.dataset_type = DatasetType.TEST
            dataset_reader.read(state)
            test_dataset = state.dataset
        else:
            # Split dataset into training and test dataset...
            x_training, x_test, y_training, y_test = train_test_split(training_dataset.x,
                                                                      training_dataset.y,
                                                                      test_size=self.test_size,
                                                                      random_state=self.random_state,
                                                                      shuffle=True)
            training_dataset = replace(training_dataset, x=x_training, y=y_training, type=DatasetType.TRAINING)
            test_dataset = replace(training_dataset, x=x_test, y=y_test, type=DatasetType.TEST)

        # Train and evaluate model...
        folding_strategy = self.folding_strategy

        for fold in folding_strategy.folds:
            yield Split(folding_strategy=folding_strategy,
                        fold=fold,
                        training_dataset=training_dataset,
                        test_dataset=test_dataset)


class CrossValidationSplitter(DatasetSplitter):
    """
    Splits the available data into training and test sets corresponding to the individual folds of a cross validation.
    """

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

    def split(self) -> Generator[Split]:
        folding_strategy = self.folding_strategy
        num_folds = folding_strategy.num_folds

        log.info(
            'Performing %s %s-fold cross validation...', 'fold ' + str(folding_strategy.first + 1) +
            (' to ' + str(folding_strategy.last) if folding_strategy.num_folds_in_subset > 1 else '')
            + ' of' if folding_strategy.is_subset else 'full', num_folds)

        # Check if predefined folds are available...
        dataset_reader = self.dataset_reader
        context = dataset_reader.input_data.context
        context.include_dataset_type = False
        context.include_fold = True

        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)

        predefined_split = all(
            dataset_reader.is_available(replace(state, fold=Fold(index=fold))) for fold in range(num_folds))

        if predefined_split:
            return self.__predefined_cross_validation()

        return self.__cross_validation()

    def __predefined_cross_validation(self) -> Generator[Split]:
        dataset_reader = self.dataset_reader
        context = dataset_reader.input_data.context
        context.include_dataset_type = False
        context.include_fold = True

        # Load (training) dataset...
        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)
        dataset_reader.read(state)
        data = [state.dataset]

        # Load datasets for the remaining folds...
        folding_strategy = self.folding_strategy
        num_folds = folding_strategy.num_folds

        for fold in range(1, num_folds):
            state = replace(state, fold=Fold(index=fold))
            dataset_reader.read(state)
            data.append(state.dataset)

        # Perform cross-validation...
        for fold in folding_strategy.folds:
            log.info('Fold %s / %s:', (fold.index + 1), num_folds)

            # Create training dataset for current fold...
            training_dataset = None

            for other_fold in range(num_folds):
                if other_fold != fold.index:
                    dataset = data[other_fold]

                    if training_dataset:
                        training_dataset.x = vstack((training_dataset.x, dataset.x))
                        training_dataset.y = vstack((training_dataset.y, dataset.y))
                    else:
                        training_dataset = replace(dataset, type=DatasetType.TRAINING)

            # Obtain test dataset for current fold...
            test_dataset = replace(data[fold.index], type=DatasetType.TEST)

            # Train and evaluate model...
            yield Split(folding_strategy=folding_strategy,
                        fold=fold,
                        training_dataset=training_dataset,
                        test_dataset=test_dataset)

    def __cross_validation(self) -> Generator[Split]:
        dataset_reader = self.dataset_reader
        context = dataset_reader.input_data.context
        context.include_dataset_type = False
        context.include_fold = False

        # Load (training) dataset...
        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)
        dataset_reader.read(state)
        dataset = state.dataset

        # Perform cross-validation...
        folding_strategy = self.folding_strategy
        num_folds = folding_strategy.num_folds
        splits = enumerate(
            KFold(n_splits=num_folds, random_state=self.random_state, shuffle=True).split(dataset.x, dataset.y))

        for fold in folding_strategy.folds:
            split_index, (training_examples, test_examples) = next(splits, (None, (None, None)))

            while split_index and split_index < fold.index:
                split_index, (training_examples, test_examples) = next(splits, (None, (None, None)))

            log.info('Fold %s / %s:', (fold.index + 1), num_folds)

            # Create training dataset for current fold...
            training_dataset = replace(dataset,
                                       x=dataset.x[training_examples],
                                       y=dataset.y[training_examples],
                                       type=DatasetType.TRAINING)

            # Create test dataset for current fold...
            test_dataset = replace(dataset,
                                   x=dataset.x[test_examples],
                                   y=dataset.y[test_examples],
                                   type=DatasetType.TEST)

            # Train and evaluate model...
            yield Split(folding_strategy=folding_strategy,
                        fold=fold,
                        training_dataset=training_dataset,
                        test_dataset=test_dataset)
