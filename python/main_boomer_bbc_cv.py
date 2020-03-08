#!/usr/bin/python

import argparse
import logging as log
from typing import List

import numpy as np

from args import optional_string, log_level, string_list, float_list, int_list, target_measure
from boomer.algorithm.model import DTYPE_FLOAT64
from boomer.algorithm.rule_learners import Boomer
from boomer.bbc_cv import BbcCv, BbcCvAdapter


class BoomerBccCvAdapter(BbcCvAdapter):

    def __init__(self, data_dir: str, data_set: str, num_folds: int, model_dir: str, min_rules:int, max_rules: int,
                 step_size_rules: int):
        super().__init__(data_dir, data_set, num_folds, model_dir)
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.step_size_rules = step_size_rules

    def _store_predictions(self, model, test_indices, test_x, num_total_examples: int, num_labels: int,
                           predictions, configurations):
        num_rules = len(model)
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
        min_rules = self.min_rules
        min_rules = max(min_rules, 1) if min_rules != -1 else 1
        max_rules = self.max_rules
        max_rules = min(num_rules, max_rules) if max_rules != -1 else num_rules
        step_size = min(max(1, self.step_size_rules), max_rules)

        for n in range(max_rules):
            rule = model.pop(0)

            if test_indices is None:
                rule.predict(test_x, current_predictions)
            else:
                masked_predictions = current_predictions[test_indices, :]
                rule.predict(test_x, masked_predictions)
                current_predictions[test_indices, :] = masked_predictions

            current_config['num_rules'] = (n + 1)

            if min_rules <= n < max_rules - 1 and (n + 1) % step_size == 0:
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


def __create_configurations(arguments) -> List[dict]:
    num_rules_values: List[int] = arguments.num_rules
    loss_values: List[str] = arguments.loss
    head_refinement_values: List[str] = [None if x.lower() == 'none' else x for x in arguments.head_refinement]
    label_sub_sampling_values: List[int] = arguments.label_sub_sampling
    instance_sub_sampling_values: List[str] = [None if x.lower() == 'none' else x for x in
                                               arguments.instance_sub_sampling]
    feature_sub_sampling_values: List[str] = [None if x.lower() == 'none' else x for x in
                                              arguments.feature_sub_sampling]
    pruning_values: List[str] = [None if x.lower() == 'none' else x for x in arguments.pruning]
    shrinkage_values: List[float] = arguments.shrinkage
    l2_regularization_weight_values: List[float] = arguments.l2_regularization_weight
    result: List[dict] = []

    for num_rules in num_rules_values:
        for loss in loss_values:
            for pruning in pruning_values:
                for instance_sub_sampling in instance_sub_sampling_values:
                    for feature_sub_sampling in feature_sub_sampling_values:
                        for shrinkage in shrinkage_values:
                            for l2_regularization_weight in l2_regularization_weight_values:
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
                                                'l2_regularization_weight': l2_regularization_weight,
                                                'head_refinement': head_refinement,
                                                'label_sub_sampling': label_sub_sampling
                                            }
                                            result.append(configuration)

    return result


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
    parser.add_argument('--num-bootstraps', type=int, default=100,
                        help='The number of bootstrap iterations to be performed')
    parser.add_argument('--min-rules', type=int, default=-1,
                        help='The minimum number of rules to be used for testing models')
    parser.add_argument('--max-rules', type=int, default=-1,
                        help='The maximum number of rules to be used for testing models')
    parser.add_argument('--step-size-rules', type=int, default=50,
                        help='The step size to be used for testing subsets of a model\'s rules')
    parser.add_argument('--target-measure', type=target_measure, default='hamming-loss',
                        help='The target measure to be used for evaluating different configurations on the tuning set')
    parser.add_argument('--num-rules', type=int_list, default='500',
                        help='The values for the parameter \'num_rules\' as a comma-separated list')
    parser.add_argument('--loss', type=string_list, default='macro-squared-error-loss',
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
    parser.add_argument('--l2-regularization-weight', type=float_list, default='1.0',
                        help='The values for the parameter \'l2-regularization-weight\' as a comma-separated list')
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    target_measure, target_measure_is_loss = args.target_measure
    base_configurations = __create_configurations(args)
    learner = Boomer()
    bbc_cv_adapter = BoomerBccCvAdapter(data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                                        model_dir=args.model_dir, min_rules=args.min_rules, max_rules=args.max_rules,
                                        step_size_rules=args.step_size_rules)
    bbc_cv = BbcCv(output_dir=args.output_dir, configurations=base_configurations, adapter=bbc_cv_adapter,
                   learner=learner)
    bbc_cv.random_state = args.random_state
    bbc_cv.evaluate(num_bootstraps=args.num_bootstraps, target_measure=target_measure,
                    target_measure_is_loss=target_measure_is_loss)
