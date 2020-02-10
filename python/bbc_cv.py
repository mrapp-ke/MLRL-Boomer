#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Implements "Bootstrap Bias Corrected Cross Validation" (BBC-CV) for evaluating different configurations of a learner and
estimating unbiased performance estimations (see https://link.springer.com/article/10.1007/s10994-018-5714-4).
"""
import argparse
import logging as log
from typing import List, Set, Tuple

import numpy as np
import sklearn.metrics as metrics
from sklearn.base import clone
from sklearn.utils import check_random_state
from skmultilearn.base import MLClassifierBase

from args import optional_string, log_level, string_list, float_list, int_list
from boomer.algorithm.model import Theory, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INTP, DTYPE_UINT8
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.rule_learners import Boomer
from boomer.evaluation import ClassificationEvaluation, EvaluationCsvOutput, EvaluationLogOutput
from boomer.io import open_writable_csv_file, create_csv_dict_writer
from boomer.training import CrossValidation


class Predictor(CrossValidation, MLClassifierBase):

    def __init__(self, data_dir: str, data_set: str, num_folds: int, model_dir: str, base_learner: Boomer,
                 configuration: dict, max_rules: int, step_size_rules: int, store_true_labels: bool):
        super().__init__(data_dir, data_set, num_folds, -1)
        self.persistence = ModelPersistence(model_dir=model_dir)
        self.base_learner = base_learner
        self.configuration = configuration
        self.predictions = []
        self.configurations = []
        self.true_labels = None
        self.store_true_labels = store_true_labels
        self.max_rules = max_rules
        self.step_size_rules = step_size_rules
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
        current_learner = clone(self.base_learner)
        current_learner.random_state = self.random_state
        current_learner.fold = current_fold
        model_name = current_learner.get_name()
        theory: Theory = self.persistence.load_model(model_name=model_name, file_name_suffix=Boomer.PREFIX_RULES,
                                                     fold=current_fold, raise_exception=True)

        num_rules = len(theory)
        predictions = self.predictions
        configurations = self.configurations
        c = 0

        if len(predictions) > c:
            current_predictions = predictions[c]
            current_config = configurations[c]
        else:
            current_predictions = np.zeros((num_total_examples, num_labels), dtype=DTYPE_FLOAT64)
            predictions.append(current_predictions)
            current_config = self.configuration.copy()
            configurations.append(current_config)

        # Store predictions...
        max_rules = self.max_rules
        max_rules = min(num_rules, max_rules) if max_rules != -1 else num_rules
        step_size = min(max(1, self.step_size_rules), max_rules)

        for n in range(max_rules):
            rule = theory.pop(0)

            if test_indices is None:
                rule.predict(test_x, current_predictions)
            else:
                masked_predictions = current_predictions[test_indices, :]
                rule.predict(test_x, masked_predictions)
                current_predictions[test_indices, :] = masked_predictions

            current_config['num_rules'] = (n + 1)

            if n < max_rules - 1 and (n + 1) % step_size == 0:
                c += 1

                if len(predictions) > c:
                    old_predictions = current_predictions
                    current_predictions = predictions[c]
                    current_predictions[test_indices] = old_predictions[test_indices]
                    current_config = configurations[c]
                else:
                    current_predictions = current_predictions.copy()
                    predictions.append(current_predictions)
                    current_config = current_config.copy()
                    configurations.append(current_config)

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


def __create_configurations(arguments) -> List[dict]:
    num_rules_values: List[int] = arguments.num_rules
    loss_values: List[str] = arguments.loss
    head_refinement_values: List[str] = arguments.head_refinement
    label_sub_sampling_values: List[int] = arguments.label_sub_sampling
    instance_sub_sampling_values: List[str] = arguments.instance_sub_sampling
    feature_sub_sampling_values: List[str] = arguments.feature_sub_sampling
    pruning_values: List[str] = arguments.pruning
    shrinkage_values: List[float] = arguments.shrinkage
    result: List[dict] = []

    for num_rules in num_rules_values:
        for loss in loss_values:
            for pruning in pruning_values:
                for instance_sub_sampling in instance_sub_sampling_values:
                    for feature_sub_sampling in feature_sub_sampling_values:
                        for shrinkage in shrinkage_values:
                            for head_refinement in head_refinement_values:
                                for label_sub_sampling in label_sub_sampling_values:
                                    if head_refinement == 'full' or label_sub_sampling == -1:
                                        configuration = {
                                            'num_rules': num_rules,
                                            'loss': loss,
                                            'pruning': pruning,
                                            'instance_sub_sampling': instance_sub_sampling,
                                            'feature_sub_sampling': feature_sub_sampling,
                                            'shrinkage': shrinkage,
                                            'head_refinement': head_refinement,
                                            'label_sub_sampling': label_sub_sampling
                                        }
                                        result.append(configuration)

    return result


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
            current_configuration = list_of_configurations[row]

            for param in parameters:
                columns[param] = current_configuration[param]

            for col in range(num_cols):
                columns['bootstrap_' + str(col + 1)] = evaluation_scores[row, col]

            csv_writer.writerow(columns)


def __write_tuning_scores_per_parameter(output_dir: str, evaluation_scores: np.ndarray, configurations: List[dict],
                                        is_gain_metric: bool):
    parameters = sorted(configurations[0].keys())
    parameter_values: dict = {}

    for current_config in configurations:
        for parameter in parameters:
            values: Set[str] = parameter_values[parameter] if parameter in parameter_values else set()
            values.add(current_config[parameter])
            parameter_values[parameter] = values

    for parameter in parameters:
        if parameter == 'head_refinement' or parameter == 'label_sub_sampling' or len(parameter_values[parameter]) < 2:
            del parameter_values[parameter]

    num_rows = evaluation_scores_tuning.shape[1]

    for parameter in parameter_values.keys():
        header = sorted(parameter_values[parameter])
        indices_dict = dict()

        for value in header:
            for config_index, current_config in enumerate(configurations):
                if current_config[parameter] == value:
                    indices = indices_dict[value] if value in indices_dict else []
                    indices.append(config_index)
                    indices_dict[value] = indices

        with open_writable_csv_file(output_dir, 'tuning_scores_per_' + str(parameter), fold=None,
                                    append=False) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)

            for row in range(num_rows):
                columns = {}

                for value in header:
                    indices = indices_dict[value]
                    scores = evaluation_scores[np.asarray(indices), row]
                    best_score = np.max(scores) if is_gain_metric else np.min(scores)
                    columns[value] = best_score

                csv_writer.writerow(columns)


def __write_tuning_scores_per_num_labels(output_dir: str, evaluation_scores: np.ndarray, configurations: List[dict],
                                         is_gain_metric: bool):
    parameter_values: Set[Tuple] = set()

    for current_config in configurations:
        key = (current_config['head_refinement'], current_config['label_sub_sampling'])
        parameter_values.add(key)

    if len(parameter_values) > 0:
        num_rows = evaluation_scores_tuning.shape[1]
        header = ['head-refinement=' + str(x[0]) + '_label-sub-sampling=' + str(x[1]) for x in parameter_values]
    
        with open_writable_csv_file(output_dir, 'tuning_scores_per_num_labels', fold=None, append=False) as csv_file:
            csv_writer = create_csv_dict_writer(csv_file, header)
            indices_dict = dict()

            for key in parameter_values:
                indices = indices_dict[key] if key in indices_dict else []

                for config_index, current_config in enumerate(configurations):
                    if current_config['head_refinement'] == key[0] and current_config['label_sub_sampling'] == key[1]:
                        indices.append(config_index)

                indices_dict[key] = indices

            for row in range(num_rows):
                columns = {}

                for key_index, key in enumerate(parameter_values):
                    indices = indices_dict[key]
                    scores = evaluation_scores[np.asarray(indices), row]
                    best_score = np.max(scores) if is_gain_metric else np.min(scores)
                    columns[header[key_index]] = best_score

                csv_writer.writerow(columns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs BBC-CV using models that have been trained using CV')
    parser.add_argument('--log-level', type=log_level, default='info', help='The log level to be used')
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    parser.add_argument('--dataset', type=str, help='The name of the data set to be used')
    parser.add_argument('--folds', type=int, default=1, help='The total number of folds to be used by cross validation')
    parser.add_argument('--model-dir', type=str, help='The path of the directory where the models are stored')
    parser.add_argument('--output-dir', type=optional_string, default=None,
                        help='The path of the directory into which results should be written')
    parser.add_argument('--num-bootstraps', type=int, default=1000,
                        help='The number of bootstrap iterations to be performed')
    parser.add_argument('--max-rules', type=int, default=-1,
                        help='The maximum number of rules to be used for testing models')
    parser.add_argument('--step-size-rules', type=int, default=50,
                        help='The step size to be used for testing subsets of a model\'s rules')
    parser.add_argument('--target-measure', type=str, default='hamming-loss',
                        help='The target measure to be used for evaluating different configurations on the tuning set')
    parser.add_argument('--num-rules', type=int_list, default='500',
                        help='The values for the parameter \'num_rules\' as a comma-separated list')
    parser.add_argument('--loss', type=string_list, default='squared-error-loss',
                        help='The values for the parameter \'loss\' as a comma-separated list')
    parser.add_argument('--head-refinement', type=string_list, default='single-label',
                        help='The values for the parameter \'head_refinement\' as a comma-separated list')
    parser.add_argument('--label-sub-sampling', type=int_list, default='-1',
                        help='The values for the parameter \'label_sub_sampling\' as a comma-separated list')
    parser.add_argument('--instance-sub-sampling', type=string_list, default='None',
                        help='The values for the parameter \'instance_sub_sampling\' as a comma-separated list')
    parser.add_argument('--feature-sub-sampling', type=string_list, default='None',
                        help='The values for the parameter \'feature_sub_sampling\' as a comma-separated list')
    parser.add_argument('--pruning', type=string_list, default='None',
                        help='The values for the parameter \'pruning\' as a comma-separated list')
    parser.add_argument('--shrinkage', type=float_list, default='1.0',
                        help='The values for the parameter \'shrinkage\' as a comma-separated list')
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    # Prepare target measure...
    if args.target_measure == 'hamming-loss':
        target_measure = metrics.hamming_loss
        gain_metric = False
    elif args.target_measure == 'subset-0-1-loss':
        target_measure = metrics.accuracy_score
        gain_metric = True
    else:
        raise ValueError('Invalid target measure given: \'' + str(args.target_measure) + '\'')

    # Create configurations...
    base_configurations = __create_configurations(args)
    num_configurations = len(base_configurations)
    log.info('%s configurations have been specified...', num_configurations)

    # Store predictions of the different models...
    random_state = args.random_state
    list_of_predictions: List[np.ndarray] = []
    list_of_configurations: List[dict] = []
    ground_truth_matrix = None
    learner = Boomer()

    for index, config in enumerate(base_configurations):
        log.info('Storing predictions of configuration %s / %s...', str(index + 1), num_configurations)
        learner.set_params(**config)
        predictor = Predictor(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                              model_dir=args.model_dir, base_learner=learner, configuration=config,
                              store_true_labels=(ground_truth_matrix is None), max_rules=args.max_rules,
                              step_size_rules=args.step_size_rules)
        predictor.random_state = random_state
        predictor.run()
        list_of_predictions.extend(predictor.predictions)
        list_of_configurations.extend(predictor.configurations)

        if ground_truth_matrix is None:
            ground_truth_matrix = predictor.true_labels

    # Create 3-dimensional prediction matrix....
    prediction_matrix = np.moveaxis(np.dstack(list_of_predictions), source=1, destination=2)
    prediction_matrix = np.where(prediction_matrix > 0, 1, 0)

    # Prepare evaluation...
    evaluation_outputs = [EvaluationLogOutput(output_individual_folds=False)]

    if args.output_dir is not None:
        evaluation_outputs.append(EvaluationCsvOutput(output_dir=args.output_dir, output_individual_folds=False))

    evaluation = ClassificationEvaluation(*evaluation_outputs)

    # Bootstrap sampling...
    num_examples = prediction_matrix.shape[0]
    num_configurations = prediction_matrix.shape[1]
    log.info('%s configurations have been evaluated...', num_configurations)
    num_bootstraps = args.num_bootstraps
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

        best_k = np.argmax(evaluation_scores_tuning[:, i]) if gain_metric else np.argmin(evaluation_scores_tuning[:, i])
        ground_truth_test = ground_truth_matrix[mask_test, :]
        predictions_test = prediction_matrix[mask_test, best_k, :]
        evaluation.evaluate('best_configuration', predictions_test, ground_truth_test, first_fold=0, current_fold=i,
                            last_fold=num_bootstraps - 1, num_folds=num_bootstraps)

    if args.output_dir is not None:
        __write_tuning_scores(args.output_dir, evaluation_scores_tuning, list_of_configurations)
        __write_tuning_scores_per_num_labels(args.output_dir, evaluation_scores_tuning, list_of_configurations,
                                             gain_metric)
        __write_tuning_scores_per_parameter(args.output_dir, evaluation_scores_tuning, list_of_configurations,
                                            gain_metric)
