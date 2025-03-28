"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating machine learning models using either cross validation or separate training
and test sets.
"""
import logging as log

from abc import ABC, abstractmethod
from functools import reduce
from os import path
from timeit import default_timer as timer
from typing import List, Optional

from scipy.sparse import vstack
from sklearn.model_selection import KFold, train_test_split

from mlrl.testbed.data import ArffMetaData, load_data_set, load_data_set_and_meta_data
from mlrl.testbed.dataset import Dataset
from mlrl.testbed.experiments.input.preprocessors import Preprocessor
from mlrl.testbed.fold import Fold
from mlrl.testbed.format import format_duration
from mlrl.testbed.util.io import SUFFIX_ARFF, SUFFIX_XML, get_file_name, get_file_name_per_fold


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


class DataSplitter(ABC):
    """
    An abstract base class for all classes that split a data set into training and test data.
    """

    class Callback(ABC):
        """
        An abstract base class for all classes that train and evaluate a model given a predefined split of the available
        data.
        """

        @abstractmethod
        def train_and_evaluate(self, fold: Fold, train_dataset: Dataset, test_dataset: Dataset):
            """
            The function that is invoked to train a model on a training set and evaluate it on a test set.

            :param fold:            The fold of the available data to be used for training and evaluating the model
            :param train_dataset:   The dataset to be used for training
            :param test_dataset:    The dataset to be used for testing
            """

    def run(self, callback: Callback):
        """
        :param callback: The callback that should be used for training and evaluating models
        """
        start_time = timer()
        self._split_data(callback)
        end_time = timer()
        run_time = end_time - start_time
        log.info('Successfully finished after %s', format_duration(run_time))

    @abstractmethod
    def _split_data(self, callback: Callback):
        """
        Must be implemented by subclasses in order to split the available data.

        :param callback: The callback that should be used for training and evaluating models
        """


def check_if_files_exist(directory: str, file_names: List[str]) -> bool:
    """
    Returns whether all given files exist or not. If some of the files are missing, an `IOError` is raised.

    :param directory:   The path to the directory where the files should be located
    :param file_names:  A list that contains the names of all files to be checked
    :return:            True, if all files exist, False, if all files are missing
    """
    missing_files = []

    for file_name in file_names:
        file = path.join(directory, file_name)

        if not path.isfile(file):
            missing_files.append(file)

    num_missing_files = len(missing_files)

    if num_missing_files == 0:
        return True
    if num_missing_files == len(file_names):
        return False
    raise IOError('The following files do not exist: '
                  + reduce(lambda aggr, missing_file: aggr +
                           (', ' if aggr else '') + '"' + missing_file + '"', missing_files, ''))


class NoSplitter(DataSplitter):
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

    def _split_data(self, callback: DataSplitter.Callback):
        log.warning('Not using separate training and test sets. The model will be evaluated on the training data...')

        # Load data set...
        data_set = self.data_set
        data_dir = data_set.data_dir
        data_set_name = data_set.data_set_name
        arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)
        xml_file_name = get_file_name(data_set_name, SUFFIX_XML)
        x, y, meta_data = load_data_set_and_meta_data(data_dir, arff_file_name, xml_file_name)

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            encoded_dataset = encoder.encode(Dataset(x, y, meta_data.features, meta_data.outputs))
            x = encoded_dataset.x
            meta_data = ArffMetaData(encoded_dataset.features, encoded_dataset.outputs, meta_data.outputs_at_start)

        # Train and evaluate model...
        fold = Fold(index=None, num_folds=1, is_last_fold=True)
        dataset = Dataset(x, y, meta_data.features, meta_data.outputs, Dataset.Type.TRAINING)
        callback.train_and_evaluate(fold, train_dataset=dataset, test_dataset=dataset)


class TrainTestSplitter(DataSplitter):
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

    def _split_data(self, callback: DataSplitter.Callback):
        log.info('Using separate training and test sets...')

        # Check if ARFF files with predefined training and test data are available...
        data_set = self.data_set
        data_set_name = data_set.data_set_name
        train_arff_file_name = get_file_name(Dataset.Type.TRAINING.get_file_name(data_set_name), SUFFIX_ARFF)
        test_arff_file_name = get_file_name(Dataset.Type.TEST.get_file_name(data_set_name), SUFFIX_ARFF)
        data_dir = data_set.data_dir
        predefined_split = check_if_files_exist(data_dir, [train_arff_file_name, test_arff_file_name])

        if not predefined_split:
            train_arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)

        # Load (training) data set...
        xml_file_name = get_file_name(data_set_name, SUFFIX_XML)
        train_x, train_y, meta_data = load_data_set_and_meta_data(data_dir, train_arff_file_name, xml_file_name)

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            encoded_dataset = encoder.encode(Dataset(train_x, train_y, meta_data.features, meta_data.outputs))
            train_x = encoded_dataset.x
            meta_data = ArffMetaData(encoded_dataset.features, encoded_dataset.outputs, meta_data.outputs_at_start)
        else:
            encoder = None

        if predefined_split:
            # Load test data set...
            test_x, test_y = load_data_set(data_dir, test_arff_file_name, meta_data)

            # Apply preprocessor, if necessary...
            if encoder:
                encoded_dataset = encoder.encode(Dataset(test_x, test_y, meta_data.features, meta_data.outputs))
                test_x = encoded_dataset.x
        else:
            # Split data set into training and test data...
            train_x, test_x, train_y, test_y = train_test_split(train_x,
                                                                train_y,
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                shuffle=True)

        # Train and evaluate model...
        fold = Fold(index=None, num_folds=1, is_last_fold=True)
        train_dataset = Dataset(train_x, train_y, meta_data.features, meta_data.outputs, Dataset.Type.TRAINING)
        test_dataset = Dataset(test_x, test_y, meta_data.features, meta_data.outputs, Dataset.Type.TEST)
        callback.train_and_evaluate(fold, train_dataset=train_dataset, test_dataset=test_dataset)


class CrossValidationSplitter(DataSplitter):
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
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.random_state = random_state

    def _split_data(self, callback: DataSplitter.Callback):
        num_folds = self.num_folds
        current_fold = self.current_fold
        log.info('Performing %s %s-fold cross validation...',
                 'full' if current_fold < 0 else 'fold ' + str(current_fold + 1) + ' of', num_folds)

        # Check if ARFF files with predefined folds are available...
        data_set = self.data_set
        data_set_name = data_set.data_set_name
        arff_file_names = [get_file_name_per_fold(data_set_name, SUFFIX_ARFF, fold) for fold in range(num_folds)]
        data_dir = data_set.data_dir
        predefined_split = check_if_files_exist(data_dir, arff_file_names)
        xml_file_name = get_file_name(data_set_name, SUFFIX_XML)

        if predefined_split:
            self.__predefined_cross_validation(callback,
                                               data_dir=data_dir,
                                               arff_file_names=arff_file_names,
                                               xml_file_name=xml_file_name,
                                               num_folds=num_folds,
                                               current_fold=current_fold)
        else:
            arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)
            self.__cross_validation(callback,
                                    data_dir=data_dir,
                                    arff_file_name=arff_file_name,
                                    xml_file_name=xml_file_name,
                                    num_folds=num_folds,
                                    current_fold=current_fold)

    def __predefined_cross_validation(self, callback: DataSplitter.Callback, data_dir: str, arff_file_names: List[str],
                                      xml_file_name: str, num_folds: int, current_fold: int):
        # Load first data set for the first fold...
        x, y, meta_data = load_data_set_and_meta_data(data_dir, arff_file_names[0], xml_file_name)

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            encoded_dataset = encoder.encode(Dataset(x, y, meta_data.features, meta_data.outputs))
            x = encoded_dataset.x
            meta_data = ArffMetaData(encoded_dataset.features, encoded_dataset.outputs, meta_data.outputs_at_start)
        else:
            encoder = None

        data = [(x, y)]

        # Load data sets for the remaining folds...
        for fold in range(1, num_folds):
            x, y = load_data_set(data_dir, arff_file_names[fold], meta_data)

            # Apply preprocessor, if necessary...
            if encoder:
                encoded_dataset = encoder.encode(Dataset(x, y, meta_data.features, meta_data.outputs))
                x = encoded_dataset.x

            data.append((x, y))

        # Perform cross-validation...
        for i in range(0 if current_fold < 0 else current_fold, num_folds if current_fold < 0 else current_fold + 1):
            log.info('Fold %s / %s:', (i + 1), num_folds)

            # Create training set for current fold...
            train_x = None
            train_y = None

            for other_fold in range(num_folds):
                if other_fold != i:
                    x, y = data[other_fold]

                    if train_x is None:
                        train_x = x
                        train_y = y
                    else:
                        train_x = vstack((train_x, x))
                        train_y = vstack((train_y, y))

            # Obtain test set for current fold...
            test_x, test_y = data[i]

            # Train and evaluate model...
            fold = Fold(index=i, num_folds=num_folds, is_last_fold=current_fold < 0 and i == num_folds - 1)
            train_dataset = Dataset(train_x, train_y, meta_data.features, meta_data.outputs, Dataset.Type.TRAINING)
            test_dataset = Dataset(test_x, test_y, meta_data.features, meta_data.outputs, Dataset.Type.TEST)
            callback.train_and_evaluate(fold, train_dataset=train_dataset, test_dataset=test_dataset)

    def __cross_validation(self, callback: DataSplitter.Callback, data_dir: str, arff_file_name: str,
                           xml_file_name: str, num_folds: int, current_fold: int):
        # Load data set...
        x, y, meta_data = load_data_set_and_meta_data(data_dir, arff_file_name, xml_file_name)

        # Apply preprocessor, if necessary...
        preprocessor = self.preprocessor

        if preprocessor:
            encoder = preprocessor.create_encoder()
            encoded_dataset = encoder.encode(Dataset(x, y, meta_data.features, meta_data.outputs))
            x = encoded_dataset.x
            meta_data = ArffMetaData(encoded_dataset.features, encoded_dataset.outputs, meta_data.outputs_at_start)

        # Perform cross-validation...
        k_fold = KFold(n_splits=num_folds, random_state=self.random_state, shuffle=True)

        for i, (train_indices, test_indices) in enumerate(k_fold.split(x, y)):
            if current_fold < 0 or i == current_fold:
                log.info('Fold %s / %s:', (i + 1), num_folds)

                # Create training set for current fold...
                train_x = x[train_indices]
                train_y = y[train_indices]

                # Create test set for current fold...
                test_x = x[test_indices]
                test_y = y[test_indices]

                # Train and evaluate model...
                fold = Fold(index=i, num_folds=num_folds, is_last_fold=current_fold < 0 and i == num_folds - 1)
                train_dataset = Dataset(train_x, train_y, meta_data.features, meta_data.outputs, Dataset.Type.TRAINING)
                test_dataset = Dataset(test_x, test_y, meta_data.features, meta_data.outputs, Dataset.Type.TEST)
                callback.train_and_evaluate(fold, train_dataset=train_dataset, test_dataset=test_dataset)
