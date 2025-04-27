"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating machine learning models using either cross validation or separate training
and test sets.
"""
import logging as log

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from os import path
from typing import Generator, Optional

from scipy.sparse import vstack
from sklearn.model_selection import KFold, train_test_split

from mlrl.common.util.format import format_iterable

from mlrl.testbed.experiments.data import FilePath
from mlrl.testbed.experiments.dataset import Dataset, DatasetType
from mlrl.testbed.experiments.fold import Fold, FoldingStrategy
from mlrl.testbed.experiments.input.dataset import InputDataset
from mlrl.testbed.experiments.input.dataset.preprocessors import Preprocessor
from mlrl.testbed.experiments.input.sources import ArffFileSource
from mlrl.testbed.experiments.output.sinks import ArffFileSink
from mlrl.testbed.experiments.problem_type import ProblemType
from mlrl.testbed.experiments.state import ExperimentState


class DataSet:
    """
    Stores the properties of a data set to be used for training and evaluating machine learning models.
    """

    def __init__(self, data_dir: str, data_set_name: str, use_one_hot_encoding: bool):
        """
        :param data_dir:                The path to the directory where the data set is located
        :param data_set_name:           The name of the data set
        :param use_one_hot_encoding:    True, if one-hot-encoding should be used to encode nominal features, False
                                        otherwise
        """
        self.data_dir = data_dir
        self.data_set_name = data_set_name
        self.use_one_hot_encoding = use_one_hot_encoding


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


def check_if_files_exist(*files: str) -> bool:
    """
    Returns whether all given files exist or not. If some of the files are missing, an `IOError` is raised.

    :param files:   The files to be checked
    :return:        True, if all files exist, False, if all files are missing
    """
    missing_files = [file for file in files if not path.isfile(file)]

    if not missing_files:
        return True
    if len(missing_files) == len(files):
        return False
    raise IOError('The following files do not exist: ' + format_iterable(missing_files, delimiter='"'))


class NoSplitter(DatasetSplitter):
    """
    Does not split the available data into separate train and test sets.
    """

    def __init__(self, data_set: DataSet, preprocessor: Optional[Preprocessor]):
        """
        :param data_set:        The properties of the data set to be used
        :param preprocessor:    An optional `Preprocessor` to be applied to the available data
        """
        self.data_set = data_set
        self.preprocessor = preprocessor
        self.folding_strategy = FoldingStrategy(num_folds=1, first=0, last=1)

    def split(self) -> Generator[Split]:
        log.warning('Not using separate training and test sets. The model will be evaluated on the training data...')

        # Load data set...
        data_set = self.data_set
        data_dir = data_set.data_dir
        data_set_name = data_set.data_set_name
        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)
        input_dataset = InputDataset(data_set_name)
        context = input_dataset.default_context
        context.include_prediction_scope = False
        context.include_dataset_type = False
        context.include_fold = False
        ArffFileSource(directory=data_dir).read_from_source(state=state, input_data=input_dataset)
        dataset = state.dataset

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            dataset = encoder.encode(dataset)

        # Train and evaluate model...
        folding_strategy = self.folding_strategy

        for fold in folding_strategy.folds:
            yield Split(folding_strategy=folding_strategy, fold=fold, training_dataset=dataset, test_dataset=dataset)


class TrainTestSplitter(DatasetSplitter):
    """
    Splits the available data into a single train and test set.
    """

    def __init__(self, data_set: DataSet, preprocessor: Optional[Preprocessor], test_size: float, random_state: int):
        """
        :param data_set:        The properties of the data set to be used
        :param preprocessor:    An optional `Preprocessor` to be applied to the train and test set
        :param test_size:       The fraction of the available data to be used as the test set
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        """
        self.data_set = data_set
        self.preprocessor = preprocessor
        self.test_size = test_size
        self.random_state = random_state
        self.folding_strategy = FoldingStrategy(num_folds=1, first=0, last=1)

    def split(self) -> Generator[Split]:
        log.info('Using separate training and test sets...')

        # Check if ARFF files with predefined training and test data are available...
        data_set = self.data_set
        data_set_name = data_set.data_set_name
        data_dir = data_set.data_dir

        input_dataset = InputDataset(data_set_name)
        context = input_dataset.default_context
        context.include_dataset_type = True
        context.include_prediction_scope = False
        context.include_fold = False

        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)

        file_path = FilePath(data_dir, data_set_name, ArffFileSink.SUFFIX_ARFF, context)
        context.include_dataset_type = check_if_files_exist(
            file_path.resolve(state),
            file_path.resolve(replace(state, dataset_type=DatasetType.TEST)),
        )

        # Load (training) data set...
        source = ArffFileSource(directory=data_dir)
        source.read_from_source(state=state, input_data=input_dataset)
        training_dataset = state.dataset

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            training_dataset = encoder.encode(training_dataset)
        else:
            encoder = None

        if context.include_dataset_type:
            # Load test data set...
            state.dataset_type = DatasetType.TEST
            source.read_from_source(state=state, input_data=input_dataset)
            test_dataset = state.dataset

            # Apply preprocessor, if necessary...
            if encoder:
                test_dataset = encoder.encode(test_dataset)
        else:
            # Split data set into training and test data...
            train_x, test_x, train_y, test_y = train_test_split(training_dataset.x,
                                                                training_dataset.y,
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                shuffle=True)
            training_dataset = replace(training_dataset, x=train_x, y=train_y, type=DatasetType.TRAINING)
            test_dataset = replace(training_dataset, x=test_x, y=test_y, type=DatasetType.TEST)

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

    def __init__(self, data_set: DataSet, preprocessor: Optional[Preprocessor], num_folds: int, current_fold: int,
                 random_state: int):
        """
        :param data_set:        The properties of the data set to be used
        :param preprocessor:    An optional `Preprocessor` to be applied to the individual folds
        :param num_folds:       The total number of folds to be used by cross validation or 1, if separate training and
                                test sets should be used
        :param current_fold:    The cross validation fold to be performed or -1, if all folds should be performed
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        """
        self.data_set = data_set
        self.preprocessor = preprocessor
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

        # Check if ARFF files with predefined folds are available...
        data_set = self.data_set
        data_set_name = data_set.data_set_name
        data_dir = data_set.data_dir

        input_dataset = InputDataset(data_set_name)
        context = input_dataset.default_context
        context.include_dataset_type = False
        context.include_prediction_scope = False
        context.include_fold = True

        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)

        file_path = FilePath(data_dir, data_set_name, ArffFileSink.SUFFIX_ARFF, context)
        predefined_split = check_if_files_exist(
            *[file_path.resolve(replace(state, fold=Fold(index=fold))) for fold in range(num_folds)])

        if predefined_split:
            return self.__predefined_cross_validation(data_dir=data_dir, dataset_name=data_set_name)

        return self.__cross_validation(data_dir=data_dir, dataset_name=data_set_name)

    def __predefined_cross_validation(self, data_dir: str, dataset_name: str) -> Generator[Split]:
        input_dataset = InputDataset(dataset_name)
        context = input_dataset.default_context
        context.include_dataset_type = False
        context.include_prediction_scope = False
        context.include_fold = True

        # Load (training) data set...
        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)
        source = ArffFileSource(directory=data_dir)
        source.read_from_source(state=state, input_data=input_dataset)
        dataset = state.dataset

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            dataset = encoder.encode(dataset)
        else:
            encoder = None

        data = [dataset]

        # Load data sets for the remaining folds...
        folding_strategy = self.folding_strategy
        num_folds = folding_strategy.num_folds

        for fold in range(1, num_folds):
            state = replace(state, fold=Fold(index=fold))
            source.read_from_source(state=state, input_data=input_dataset)
            dataset = state.dataset

            # Apply preprocessor, if necessary...
            if encoder:
                dataset = encoder.encode(dataset)

            data.append(dataset)

        # Perform cross-validation...
        for fold in folding_strategy.folds:
            log.info('Fold %s / %s:', (fold.index + 1), num_folds)

            # Create training set for current fold...
            training_dataset = None

            for other_fold in range(num_folds):
                if other_fold != fold.index:
                    dataset = data[other_fold]

                    if training_dataset:
                        training_dataset.x = vstack((training_dataset.x, dataset.x))
                        training_dataset.y = vstack((training_dataset.y, dataset.y))
                    else:
                        training_dataset = replace(dataset, type=DatasetType.TRAINING)

            # Obtain test set for current fold...
            test_dataset = replace(data[fold.index], type=DatasetType.TEST)

            # Train and evaluate model...
            yield Split(folding_strategy=folding_strategy,
                        fold=fold,
                        training_dataset=training_dataset,
                        test_dataset=test_dataset)

    def __cross_validation(self, data_dir: str, dataset_name: str) -> Generator[Split]:
        input_dataset = InputDataset(dataset_name)
        context = input_dataset.default_context
        context.include_dataset_type = False
        context.include_prediction_scope = False
        context.include_fold = False

        # Load (training) data set...
        state = ExperimentState(problem_type=ProblemType.CLASSIFICATION,
                                folding_strategy=self.folding_strategy,
                                fold=Fold(index=0),
                                dataset_type=DatasetType.TRAINING)
        source = ArffFileSource(directory=data_dir)
        source.read_from_source(state=state, input_data=input_dataset)
        dataset = state.dataset

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            dataset = encoder.encode(dataset)

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

            # Create training set for current fold...
            training_dataset = replace(dataset,
                                       x=dataset.x[training_examples],
                                       y=dataset.y[training_examples],
                                       type=DatasetType.TRAINING)

            # Create test set for current fold...
            test_dataset = replace(dataset,
                                   x=dataset.x[test_examples],
                                   y=dataset.y[test_examples],
                                   type=DatasetType.TEST)

            # Train and evaluate model...
            yield Split(folding_strategy=folding_strategy,
                        fold=fold,
                        training_dataset=training_dataset,
                        test_dataset=test_dataset)
