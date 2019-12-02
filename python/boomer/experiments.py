#!/usr/bin/python


"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for training and evaluating multi-label classifiers using either cross validation or separate training
and test sets.
"""
import logging as log
import os.path as path
from abc import abstractmethod
from timeit import default_timer as timer

from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

from boomer.data import load_data_set_and_meta_data, load_data_set, one_hot_encode
from boomer.evaluation import Evaluation
from boomer.learners import Randomized, MLLearner, BatchMLLearner


class CrossValidation(Randomized):
    """
    A base class for all classes that use cross validation or a train-test split to train and evaluate a multi-label
    classifier or ranker.
    """

    def __init__(self, data_dir: str, data_set: str, folds: int):
        """
        :param data_dir:    The path of the directory that contains the .arff file(s)
        :param data_set:    Name of the data set, e.g. "emotions"
        :param folds:       Number of folds to be used by cross validation or 1, if separate training and test sets
                            should be used
        """
        self.data_dir = data_dir
        self.data_set = data_set
        self.folds = folds

    def run(self):
        start_time = timer()

        if self.folds > 1:
            self.__cross_validate()
        else:
            self.__train_test_split()

        end_time = timer()
        run_time = end_time - start_time
        log.info('Successfully finished after %s seconds', run_time)

    def __cross_validate(self):
        """
        Performs n-fold cross validation.
        """

        log.info('Performing %s-fold cross validation...', self.folds)
        x, y, meta_data = load_data_set_and_meta_data(self.data_dir, self.data_set + ".arff", self.data_set + ".xml")
        x, _ = one_hot_encode(x, y, meta_data.nominal_attributes)

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

        # Load training data
        train_arff_file_name = self.data_set + '-train.arff'
        train_arff_file = path.join(self.data_dir, train_arff_file_name)
        test_data_exists = True

        if not path.isfile(train_arff_file):
            train_arff_file_name = self.data_set + '.arff'
            log.warning('File \'' + train_arff_file + '\' does not exist. Using \'' +
                        path.join(self.data_dir, train_arff_file_name) + '\' instead!')
            test_data_exists = False

        train_x, train_y, meta_data = load_data_set_and_meta_data(self.data_dir, train_arff_file_name,
                                                                  self.data_set + '.xml')
        train_x, encoder = one_hot_encode(train_x, train_y, meta_data.nominal_attributes)

        # Load test data
        if test_data_exists:
            test_x, test_y = load_data_set(self.data_dir, self.data_set + '-test.arff', meta_data)
            test_x, _ = one_hot_encode(test_x, test_y, meta_data.nominal_attributes, encoder=encoder)
        else:
            log.warning('No test data set available. Model will be evaluated on the training data!')
            test_x = train_x
            test_y = train_y

        # Train and evaluate classifier
        self._train_and_evaluate(train_x, train_y, test_x, test_y, current_fold=0, total_folds=1)

    @abstractmethod
    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, current_fold: int, total_folds: int):
        """
        The function that is invoked to build a multi-label classifier or ranker on a training set and evaluate it on a
        test set.

        :param train_x:         The feature matrix of the training examples
        :param train_y:         The label matrix of the training examples
        :param test_x:          The feature matrix of the test examples
        :param test_y:          The label matrix of the test examples
        :param current_fold:    The current fold starting at 0
        :param total_folds:     The total number of folds or 0, if no cross validation is used
        """
        pass


class AbstractExperiment(CrossValidation):
    """
    An abstract base class for all experiments. It automatically encodes nominal attributes using one-hot encoding.
    """

    def __init__(self, evaluation: Evaluation, data_dir: str, data_set: str, folds: int = 1):
        """
        :param evaluation:      The evaluation to be used
        """

        super().__init__(data_dir, data_set, folds)
        self.evaluation = evaluation

    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, current_fold: int, total_folds: int):
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


class BatchExperiment(AbstractExperiment):
    """
    An experiment that trains and evaluates several variants of a multi-label classifier or ranker on the same data set
    using cross validation or separate training and test sets.
    """

    def __init__(self, name: str, learner: BatchMLLearner, evaluation: Evaluation, data_dir: str, data_set: str,
                 folds: int = 1):
        """
        :param name:    The name of the experiment that uses the default variant
        :param learner: The default variant of the scikit-learn classifier to be trained
        """
        super().__init__(evaluation, data_dir, data_set, folds)
        self.variants = [({}, name)]
        self.learner = learner

    def add_variant(self, name: str, **kwargs):
        """
        Adds a new variant to the batch experiment.

        :param name:    The name of the variant
        :param kwargs:  The arguments to be passed to the classifier when creating a copy of the current variant
        """
        self.variants.append((kwargs, name))

    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, current_fold: int, total_folds: int):
        next_variant = self.learner

        for args, name in self.variants:
            log.info('Starting experiment \"' + name + '\"...')
            next_variant = next_variant.copy_classifier(**args)
            next_variant.random_state = self.random_state
            next_variant.fold = current_fold

            try:
                # Obtain predictions without re-training the classifier, if possible
                predictions = next_variant.predict(test_x)
            except NotFittedError:
                # Train classifier and obtain predictions
                if next_variant.fold == current_fold:
                    try:
                        next_variant.partial_fit(train_x, train_y)
                    except NotFittedError:
                        next_variant.fit(train_x, train_y)
                else:
                    next_variant.fit(train_x, train_y)

                predictions = next_variant.predict(test_x)

            # Evaluate predictions
            self.evaluation.evaluate(name, predictions, test_y, current_fold=current_fold, total_folds=self.folds)
