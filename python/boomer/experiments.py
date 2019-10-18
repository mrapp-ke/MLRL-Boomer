#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for training and evaluating multi-label classifiers using either cross validation or separate training
and test sets.
"""
import logging as log
import os.path as path
from abc import abstractmethod

from sklearn.model_selection import KFold
from skmultilearn.dataset import load_from_arff

from boomer.data import one_hot_encode
from boomer.data import parse_metadata
from boomer.evaluation import Evaluation
from boomer.learners import Randomized, MLLearner


class AbstractExperiment(Randomized):
    """
    An abstract base class for all experiments. It automatically encodes nominal attributes using one-hot encoding.
    """

    def __init__(self, evaluation: Evaluation, data_dir: str, data_set: str, folds: int = 1):
        """
        :param evaluation:      The evaluation to be used
        :param data_dir:        The path of the directory that contains the .arff file(s)
        :param data_set:        Name of the data set, e.g. "emotions".
        :param folds:           Number of folds to be used by cross validation or 1, if separate training and test sets
                                should be used
        """

        self.evaluation = evaluation
        self.data_dir = data_dir
        self.data_set = data_set
        self.folds = folds

    def run(self):
        if self.folds > 1:
            self.__cross_validate()
        else:
            self.__train_test_split()

    def __cross_validate(self):
        """
        Trains the classifier used in the experiment using n-fold cross validation.
        """

        log.info('Performing %s-fold cross validation...', self.folds)

        log.debug('Parsing meta data from data set...')
        arff_file = path.join(self.data_dir, self.data_set + '.arff')
        xml_file = path.join(self.data_dir, self.data_set + '.xml')
        num_labels, label_location, nominal_attributes = parse_metadata(arff_file, xml_file)
        log.debug('Loading training set from file \"%s\"...', arff_file)
        x, y = load_from_arff(arff_file, label_count=num_labels, label_location=label_location)
        x, _ = one_hot_encode(x, y, nominal_attributes)

        # Cross validate
        k_fold = KFold(n_splits=self.folds, random_state=self.random_state, shuffle=True)
        current_fold = 0

        for train, test in k_fold.split(x, y):
            log.info('Fold %s / %s:', (current_fold + 1), self.folds)

            # Create training set for current fold
            train_x = x[train]
            train_y = y[train]

            # Create test set for current fold
            test_x = x[test]
            test_y = y[test]

            # Train & evaluate classifier
            self._train_and_evaluate(train_x, train_y, test_x, test_y, current_fold=current_fold,
                                     total_folds=self.folds)

            current_fold += 1

    def __train_test_split(self):
        """
        Trains the classifier used in the experiment on a training set and validates it on a test set.
        """

        log.info('Using separate training and test sets...')

        log.debug('Parsing meta data from data set...')
        train_file = path.join(self.data_dir, self.data_set + '-train.arff')
        xml_file = path.join(self.data_dir, self.data_set + '.xml')
        num_labels, label_location, nominal_attributes = parse_metadata(train_file, xml_file)

        # Load training data
        log.debug('Loading training set from file \"%s\"...', train_file)
        train_x, train_y = load_from_arff(train_file, label_count=num_labels, label_location=label_location)
        train_x, encoder = one_hot_encode(train_x, train_y, nominal_attributes)

        # Load test data
        test_file = path.join(self.data_dir, self.data_set + '-test.arff')
        log.debug('Loading test set from file \"%s\"...', test_file)
        test_x, test_y = load_from_arff(test_file, label_count=num_labels, label_location=label_location)
        test_x = one_hot_encode(test_x, test_y, nominal_attributes, encoder=encoder)

        # Train and evaluate classifier
        self._train_and_evaluate(train_x, train_y, test_x, test_y, current_fold=0, total_folds=1)

    @abstractmethod
    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, current_fold: int, total_folds: int):
        """

        :param train_x:         The feature matrix of the training examples
        :param train_y:         The label matrix of the training examples
        :param test_x:          The feature matrix of the test examples
        :param test_y:          The label matrix of the test examples
        :param current_fold:    The current fold starting at 0
        :param total_folds:     The total number of folds or 0, if no cross validation is used
        """
        pass


class Experiment(AbstractExperiment):
    """
    An experiment that trains and evaluates a single multi-label classifier or ranker on a specific data set using cross
    validation or separate training and test sets.
    """

    def __init__(self, name: str, learner: MLLearner, evaluation: Evaluation, data_dir: str, data_set: str,
                 folds: int = 1):
        """
        :param name:    The name of the experiment to be written to output files
        :param learner: The classifier or ranker to be trained
        """
        super().__init__(evaluation, data_dir, data_set, folds)
        self.name = name
        self.learner = learner

    def run(self):
        log.info('Starting experiment \"' + self.name + '\"...')
        super().run()

    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, current_fold: int, total_folds: int):
        # Train classifier
        self.learner.random_state = self.random_state
        self.learner.fold = current_fold
        self.learner.fit(train_x, train_y)

        # Obtain and evaluate predictions for test data
        predictions = self.learner.predict(test_x)
        self.evaluation.evaluate(self.name, predictions, test_y, current_fold=current_fold, total_folds=self.folds)
