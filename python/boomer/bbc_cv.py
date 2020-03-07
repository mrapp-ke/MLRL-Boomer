#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Implements "Bootstrap Bias Corrected Cross Validation" (BBC-CV) for evaluating different configurations of a learner and
estimating unbiased performance estimations (see https://link.springer.com/article/10.1007/s10994-018-5714-4).
"""
import logging as log
from abc import abstractmethod
from typing import List

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_random_state
from skmultilearn.base import MLClassifierBase

from boomer.algorithm.model import DTYPE_INTP, DTYPE_UINT8, DTYPE_FLOAT32
from boomer.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from boomer.interfaces import Randomized
from boomer.io import open_writable_csv_file, create_csv_dict_writer
from boomer.learners import MLLearner
from boomer.persistence import ModelPersistence
from boomer.training import CrossValidation


class BbcCvAdapter(CrossValidation, MLClassifierBase):
    """
    An adapter that must be implemented for each type of model to be used with BBC-CV to obtain predictions for given
    test examples.
    """

    def __init__(self, data_dir: str, data_set: str, num_folds: int, model_dir: str):
        """
        :param model_dir: The path of the directory where the models are stored
        """
        super().__init__(data_dir, data_set, num_folds, -1)
        self.persistence = ModelPersistence(model_dir=model_dir)
        self.learner = None
        self.configuration = None
        self.store_true_labels = True
        self.require_dense = [True, True]

    def _train_and_evaluate(self, train_indices, train_x, train_y, test_indices, test_x, test_y, first_fold: int,
                            current_fold: int, last_fold: int, num_folds: int):
        num_total_examples = test_x.shape[0] + (0 if test_indices is None else train_x.shape[0])
        num_labels = test_y.shape[1]

        # Create a dense representation of the test data
        test_x = np.asfortranarray(self._ensure_input_format(test_x), dtype=DTYPE_FLOAT32)
        test_y = self._ensure_input_format(test_y)

        # Update true labels, if necessary...
        if self.store_true_labels:
            true_labels = self.true_labels

            if true_labels is None:
                if test_indices is None:
                    true_labels = test_y
                else:
                    true_labels = np.empty((num_total_examples, num_labels), dtype=DTYPE_UINT8)

                self.true_labels = true_labels

            if test_indices is not None:
                true_labels[test_indices] = test_y

        # Load theory...
        current_learner = clone(self.learner)
        current_learner.set_params(**self.configuration)
        current_learner.random_state = self.random_state
        current_learner.fold = current_fold
        model_name = current_learner.get_name()
        file_name_suffix = current_learner.get_model_prefix()
        model = self.persistence.load_model(model_name=model_name, file_name_suffix=file_name_suffix, fold=current_fold,
                                            raise_exception=True)

        predictions = self.predictions
        configurations = self.configurations
        self._store_predictions(model, test_indices, test_x, num_total_examples, num_labels, predictions,
                                configurations)

    def run(self):
        self.predictions = []
        self.configurations = []
        self.true_labels = None
        super().run()

    @abstractmethod
    def _store_predictions(self, model, test_indices, test_x, num_total_examples: int, num_labels: int, predictions,
                           configurations):
        """
        Must be implemented by subclasses to store the predictions provided by a specific model for the given test
        examples. The predictions, together with the corresponding configuration, must be stored in the given lists
        `predictions` and `configurations`. It is possible to evaluate more than one configurations by modifying the
        given model accordingly.

        :param model:               The model that should be used to make predictions
        :param test_indices:        The indices of the test examples
        :param test_x:              An array of dtype `float`, shape `(num_examples, num_features)`, representing the
                                    features of the test examples
        :param num_total_examples:  The total number of examples
        :param num_labels:          The number of labels
        :param predictions:         The list that should be used to store predictions
        :param configurations:      The list that should be used to store configurations
        """
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


class BbcCv(Randomized):
    """
    An implementation of "Bootstrap Bias Corrected Cross Validation" (BBC-CV).
    """

    def __init__(self, output_dir: str, configurations: List[dict], adapter: BbcCvAdapter, learner: MLLearner):
        """
        :param output_dir:      The path of the directory where output files should be stored
        :param configurations:  A list that contains the configurations to be evaluated
        :param adapter:         The `BbcCvAdapter` to be used
        :param learner:         The learner to be evaluated
        """
        super().__init__()
        self.output_dir = output_dir
        self.configurations = configurations
        self.adapter = adapter
        self.learner = learner

    @staticmethod
    def __write_tuning_scores(output_dir: str, evaluation_scores: np.ndarray, configurations: List[dict]):
        parameters = sorted(configurations[0].keys())
        header = list(parameters)
        num_rows = evaluation_scores.shape[0]
        num_cols = evaluation_scores.shape[1]

        for col in range(num_cols):
            header.append('bootstrap_' + str(col + 1))

        with open_writable_csv_file(output_dir, 'tuning_scores', fold=None, append=False) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)

            for row in range(num_rows):
                columns: dict = {}
                current_configuration = configurations[row]

                for param in parameters:
                    columns[param] = current_configuration[param]

                for col in range(num_cols):
                    columns['bootstrap_' + str(col + 1)] = evaluation_scores[row, col]

                csv_writer.writerow(columns)

    def evaluate(self, num_bootstraps: int, target_measure, target_measure_is_loss: bool):
        """
        :param num_bootstraps:          The number of bootstrap iterations to be performed
        :param target_measure:          The target measure to be used for parameter tuning
        :param target_measure_is_loss:  True, if the target measure is a loss, False otherwise
        """
        configurations = self.configurations
        num_configurations = len(configurations)
        log.info('%s configurations have been specified...', num_configurations)

        # Store predictions of the different models...
        random_state = self.random_state
        adapter = self.adapter
        adapter.random_state = random_state
        adapter.learner = self.learner
        list_of_predictions: List[np.ndarray] = []
        list_of_configurations: List[dict] = []
        ground_truth_matrix = None

        for index, config in enumerate(configurations):
            log.info('Storing predictions of configuration %s / %s...', str(index + 1), num_configurations)

            adapter.configuration = config
            adapter.store_true_labels = ground_truth_matrix is None
            adapter.run()

            list_of_predictions.extend(adapter.predictions)
            list_of_configurations.extend(adapter.configurations)

            if ground_truth_matrix is None:
                ground_truth_matrix = adapter.true_labels

        # Create 3-dimensional prediction matrix....
        prediction_matrix = np.moveaxis(np.dstack(list_of_predictions), source=1, destination=2)
        prediction_matrix = np.where(prediction_matrix > 0, 1, 0)

        # Prepare evaluation...
        evaluation_outputs = [EvaluationLogOutput(output_individual_folds=False)]
        output_dir = self.output_dir

        if output_dir is not None:
            evaluation_outputs.append(EvaluationCsvOutput(output_dir=output_dir, output_individual_folds=False))

        evaluation = ClassificationEvaluation(*evaluation_outputs)

        # Bootstrap sampling...
        num_examples = prediction_matrix.shape[0]
        num_configurations = prediction_matrix.shape[1]
        log.info('%s configurations have been evaluated...', num_configurations)
        bootstrapped_indices = np.empty(num_examples, dtype=DTYPE_INTP)
        mask_test = np.empty(num_examples, dtype=np.bool)
        evaluation_scores_tuning = np.empty((num_configurations, num_bootstraps), dtype=float)
        rng = check_random_state(random_state)
        rng_randint = rng.randint

        for i in range(num_bootstraps):
            mask_test[:] = True
            log.info('Sampling bootstrap examples %s / %s...', (i + 1), num_bootstraps)

            for j in range(num_examples):
                index = rng_randint(num_examples)
                bootstrapped_indices[j] = index
                mask_test[index] = False

            ground_truth_tuning = ground_truth_matrix[bootstrapped_indices, :]

            for k in range(num_configurations):
                predictions_tuning = prediction_matrix[bootstrapped_indices, k, :]
                evaluation_scores_tuning[k, i] = target_measure(ground_truth_tuning, predictions_tuning)

            best_k = np.argmin(evaluation_scores_tuning[:, i]) if target_measure_is_loss else np.argmax(
                evaluation_scores_tuning[:, i])
            ground_truth_test = ground_truth_matrix[mask_test, :]
            predictions_test = prediction_matrix[mask_test, best_k, :]
            evaluation.evaluate('best_configuration', predictions_test, ground_truth_test, first_fold=0, current_fold=i,
                                last_fold=num_bootstraps - 1, num_folds=num_bootstraps)

        # Write output files...
        if output_dir is not None:
            BbcCv.__write_tuning_scores(output_dir, evaluation_scores_tuning, list_of_configurations)
