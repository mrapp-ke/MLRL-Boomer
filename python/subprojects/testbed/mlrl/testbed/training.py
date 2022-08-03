"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for training and evaluating multi-label classifiers using either cross validation or separate training
and test sets.
"""
import logging as log
import os.path as path
from abc import ABC, abstractmethod
from enum import Enum
from timeit import default_timer as timer
from typing import Optional

from mlrl.testbed.data import MetaData, load_data_set_and_meta_data, load_data_set, one_hot_encode
from mlrl.testbed.io import SUFFIX_ARFF, SUFFIX_XML, get_file_name
from sklearn.model_selection import KFold


class DataSet:
    """
    Stores the properties of a data set to be used for training and evaluating multi-label classifiers.
    """

    def __init__(self, data_dir: str, data_set_name: str, use_one_hot_encoding: bool):
        """
        :param data_dir:                The path of the directory where the data set is located
        :param data_set_name:           The name of the data set
        :param use_one_hot_encoding:    True, if one-hot-encoding should be used to encode nominal attributes, False
                                        otherwise
        """
        self.data_dir = data_dir
        self.data_set_name = data_set_name
        self.use_one_hot_encoding = use_one_hot_encoding


class DataPartition(ABC):
    """
    Provides information about a partition of the available data that is used for training and testing.
    """

    @abstractmethod
    def get_num_folds(self) -> int:
        """
        Returns the total number of cross validation folds.

        :return: The total number of cross validation folds or 1, if no cross validation is used
        """
        pass

    @abstractmethod
    def get_fold(self) -> Optional[int]:
        """
        Returns the cross validation fold, the partition of data corresponds to.

        :return: The cross validation fold, starting at 0, or None, if no cross validation is used
        """
        pass

    def is_cross_validation_used(self) -> bool:
        """
        Returns whether cross validation is used or not.

        :return: True, if cross validation is used, False otherwise
        """
        return self.get_num_folds() > 1

    def is_last_fold(self) -> bool:
        """
        Returns whether this is the last fold or not.

        :return: True, if this is the last fold, False otherwise
        """
        return not self.is_cross_validation_used() or self.get_fold() == self.get_num_folds() - 1


class TrainingTestSplit(DataPartition):
    """
    Provides information about a predefined partition of the available data into training and test data.
    """

    def get_num_folds(self) -> int:
        return 1

    def get_fold(self) -> Optional[int]:
        return None


class CrossValidationFold(DataPartition):
    """
    Provides information a partition of the available data that is used by a single fold of a cross validation.
    """

    def __init__(self, num_folds: int, fold: int):
        """
        :param num_folds:   The total number of folds
        :param fold:        The fold, starting at 0
        """
        self.num_folds = num_folds
        self.fold = fold

    def get_num_folds(self) -> int:
        return self.num_folds

    def get_fold(self) -> Optional[int]:
        return self.fold


class DataType(Enum):
    """
    Characterizes data as either training or test data.
    """
    TRAINING = 'training'
    TEST = 'test'


class DataSplitter(ABC):
    """
    A base class for all classes that split a data set into training and test data.
    """

    def __init__(self, data_set: DataSet, num_folds: int, current_fold: int, random_state: int):
        """
        :param data_set:        The properties of the data set to be used
        :param num_folds:       The total number of folds to be used by cross validation or 1, if separate training and
                                test sets should be used
        :param current_fold:    The cross validation fold to be performed or -1, if all folds should be performed
        :param random_state:    The seed to be used by RNGs. Must be at least 1
        """
        self.data_set = data_set
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.random_state = random_state

    def run(self):
        start_time = timer()
        num_folds = self.num_folds

        if num_folds > 1:
            self.__cross_validate(num_folds)
        else:
            self.__train_test_split()

        end_time = timer()
        run_time = end_time - start_time
        log.info('Successfully finished after %s seconds', run_time)

    def __cross_validate(self, num_folds: int):
        """
        Performs n-fold cross validation.

        :param num_folds: The total number of cross validation folds
        """
        current_fold = self.current_fold
        log.info('Performing ' + (
            'full' if current_fold < 0 else ('fold ' + str(current_fold + 1) + ' of')) + ' %s-fold cross validation...',
                 num_folds)
        data_set = self.data_set
        data_set_name = data_set.data_set_name
        x, y, meta_data = load_data_set_and_meta_data(data_set.data_dir, get_file_name(data_set_name, SUFFIX_ARFF),
                                                      get_file_name(data_set_name, SUFFIX_XML))

        if data_set.use_one_hot_encoding:
            x, _, encoded_meta_data = one_hot_encode(x, y, meta_data)
        else:
            encoded_meta_data = None

        # Cross validate
        i = 0
        k_fold = KFold(n_splits=num_folds, random_state=self.random_state, shuffle=True)

        for train_indices, test_indices in k_fold.split(x, y):
            if current_fold < 0 or i == current_fold:
                log.info('Fold %s / %s:', (i + 1), num_folds)

                # Create training set for current fold
                train_x = x[train_indices]
                train_y = y[train_indices]

                # Create test set for current fold
                test_x = x[test_indices]
                test_y = y[test_indices]

                # Train & evaluate classifier
                data_partition = CrossValidationFold(num_folds=num_folds, fold=i)
                self._train_and_evaluate(encoded_meta_data if encoded_meta_data is not None else meta_data,
                                         data_partition, train_indices, train_x, train_y, test_indices, test_x, test_y)

            i += 1

    def __train_test_split(self):
        """
        Trains the classifier used in the experiment on a training set and validates it on a test set.
        """

        log.info('Using separate training and test sets...')

        # Load training data
        data_set = self.data_set
        data_dir = data_set.data_dir
        data_set_name = data_set.data_set_name
        use_one_hot_encoding = data_set.use_one_hot_encoding
        train_arff_file_name = get_file_name(data_set_name + '-train', SUFFIX_ARFF)
        train_arff_file = path.join(data_dir, train_arff_file_name)
        test_data_exists = True

        if not path.isfile(train_arff_file):
            train_arff_file_name = get_file_name(data_set_name, SUFFIX_ARFF)
            log.warning('File \'' + train_arff_file + '\' does not exist. Using \'' +
                        path.join(data_dir, train_arff_file_name) + '\' instead!')
            test_data_exists = False

        train_x, train_y, meta_data = load_data_set_and_meta_data(data_dir, train_arff_file_name,
                                                                  get_file_name(data_set_name, SUFFIX_XML))

        if use_one_hot_encoding:
            train_x, encoder, encoded_meta_data = one_hot_encode(train_x, train_y, meta_data)
        else:
            encoder = None
            encoded_meta_data = None

        # Load test data
        if test_data_exists:
            test_x, test_y = load_data_set(data_dir, get_file_name(data_set_name + '-test', SUFFIX_ARFF), meta_data)

            if encoder is not None:
                test_x, _, _ = one_hot_encode(test_x, test_y, meta_data, encoder=encoder)
        else:
            log.warning('No test data set available. Model will be evaluated on the training data!')
            test_x = train_x
            test_y = train_y

        # Train and evaluate classifier
        data_partition = TrainingTestSplit()
        self._train_and_evaluate(encoded_meta_data if encoded_meta_data is not None else meta_data, data_partition,
                                 None, train_x, train_y, None, test_x, test_y)

    @abstractmethod
    def _train_and_evaluate(self, meta_data: MetaData, data_partition: DataPartition, train_indices, train_x, train_y,
                            test_indices, test_x, test_y):
        """
        The function that is invoked to build a multi-label classifier or ranker on a training set and evaluate it on a
        test set.

        :param meta_data:       The meta-data of the training data set
        :param data_partition:  Information about the partition of data that should be used for building and evaluating
                                a classifier or ranker
        :param train_indices:   The indices of the training examples or None, if no cross validation is used
        :param train_x:         The feature matrix of the training examples
        :param train_y:         The label matrix of the training examples
        :param test_indices:    The indices of the test examples or None, if no cross validation is used
        :param test_x:          The feature matrix of the test examples
        :param test_y:          The label matrix of the test examples
        """
        pass
