"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.boosting.boosting_learners import Boomer, STATISTIC_FORMAT_VALUES, DEFAULT_RULE_VALUES, HEAD_TYPE_VALUES, \
    LABEL_BINNING_VALUES, LOSS_VALUES, CLASSIFICATION_PREDICTOR_VALUES, \
    PROBABILITY_PREDICTOR_VALUES, PARALLEL_VALUES, FEATURE_BINNING_VALUES
from mlrl.common.rule_learners import AUTOMATIC
from mlrl.common.strings import format_dict_keys, format_string_set
from mlrl.testbed.args import add_rule_learner_arguments, add_max_rules_argument, add_time_limit_argument, \
    add_label_sampling_argument, add_instance_sampling_argument, add_feature_sampling_argument, \
    add_partition_sampling_argument, add_early_stopping_argument, add_pruning_argument, add_rule_induction_argument, \
    add_parallel_prediction_argument, PARAM_FEATURE_BINNING, PARAM_HEAD_TYPE, PARAM_PARALLEL_RULE_REFINEMENT, \
    PARAM_PARALLEL_STATISTIC_UPDATE
from mlrl.testbed.args_boosting import add_shrinkage_argument, add_regularization_arguments, PARAM_STATISTIC_FORMAT, \
    PARAM_DEFAULT_RULE, PARAM_LABEL_BINNING, PARAM_LOSS, PARAM_CLASSIFICATION_PREDICTOR, PARAM_PROBABILITY_PREDICTOR
from mlrl.testbed.runnables import RuleLearnerRunnable


class BoomerRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return Boomer(random_state=args.random_state,
                      feature_format=args.feature_format,
                      label_format=args.label_format,
                      predicted_label_format=args.predicted_label_format,
                      statistic_format=args.statistic_format,
                      default_rule=args.default_rule,
                      rule_induction=args.rule_induction,
                      max_rules=args.max_rules,
                      time_limit=args.time_limit,
                      early_stopping=args.early_stopping,
                      loss=args.loss,
                      classification_predictor=args.classification_predictor,
                      probability_predictor=args.probability_predictor,
                      pruning=args.pruning,
                      label_sampling=args.label_sampling,
                      instance_sampling=args.instance_sampling,
                      shrinkage=args.shrinkage,
                      feature_sampling=args.feature_sampling,
                      holdout=args.holdout,
                      feature_binning=args.feature_binning,
                      label_binning=args.label_binning,
                      head_type=args.head_type,
                      l1_regularization_weight=args.l1_regularization_weight,
                      l2_regularization_weight=args.l2_regularization_weight,
                      parallel_rule_refinement=args.parallel_rule_refinement,
                      parallel_statistic_update=args.parallel_statistic_update,
                      parallel_prediction=args.parallel_prediction)

    def _get_learner_name(self) -> str:
        return "boomer"


def __add_arguments(parser: ArgumentParser):
    add_rule_learner_arguments(parser)
    add_max_rules_argument(parser)
    add_time_limit_argument(parser)
    add_label_sampling_argument(parser)
    add_instance_sampling_argument(parser)
    add_feature_sampling_argument(parser)
    add_partition_sampling_argument(parser)
    add_early_stopping_argument(parser)
    add_pruning_argument(parser)
    add_rule_induction_argument(parser)
    add_parallel_prediction_argument(parser)
    add_shrinkage_argument(parser)
    add_regularization_arguments(parser)
    parser.add_argument(PARAM_STATISTIC_FORMAT, type=str,
                        help='The format to be used for the representation of gradients and Hessians. Must be one of '
                             + format_string_set(STATISTIC_FORMAT_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable format is chosen automatically based on the parameters ' + PARAM_LOSS + ', '
                             + PARAM_HEAD_TYPE + ', ' + PARAM_DEFAULT_RULE + ' and the characteristics of the label '
                             + 'matrix.')
    parser.add_argument(PARAM_DEFAULT_RULE, type=str,
                        help='Whether a default rule should be induced or not. Must be one of '
                             + format_string_set(DEFAULT_RULE_VALUES) + '.')
    parser.add_argument(PARAM_FEATURE_BINNING, type=str,
                        help='The name of the strategy to be used for feature binning. Must be one of '
                             + format_dict_keys(FEATURE_BINNING_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable strategy is chosen automatically based on the characteristics of the feature '
                             + 'matrix. For additional options refer to the documentation.')
    parser.add_argument(PARAM_LABEL_BINNING, type=str,
                        help='The name of the strategy to be used for gradient-based label binning (GBLB). Must be one '
                             + 'of ' + format_dict_keys(LABEL_BINNING_VALUES) + '. If set to "' + AUTOMATIC + '", the '
                             + 'most suitable strategy is chosen automatically based on the parameters ' + PARAM_LOSS
                             + ' and ' + PARAM_HEAD_TYPE + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_LOSS, type=str,
                        help='The name of the loss function to be minimized during training. Must be one of '
                             + format_string_set(LOSS_VALUES) + '.')
    parser.add_argument(PARAM_CLASSIFICATION_PREDICTOR, type=str,
                        help='The name of the strategy to be used for predicting binary labels. Must be one of '
                             + format_string_set(CLASSIFICATION_PREDICTOR_VALUES) + '. If set to "' + AUTOMATIC + '", '
                             + 'the most suitable strategy is chosen automatically based on the parameter '
                             + PARAM_LOSS + '.')
    parser.add_argument(PARAM_PROBABILITY_PREDICTOR, type=str,
                        help='The name of the strategy to be used for predicting probabilities. Must be one of '
                             + format_string_set(PROBABILITY_PREDICTOR_VALUES) + '. If set to "' + AUTOMATIC + '", the '
                             + 'most suitable strategy is chosen automatically based on the parameter ' + PARAM_LOSS
                             + '.')
    parser.add_argument(PARAM_HEAD_TYPE, type=str,
                        help='The type of the rule heads that should be used. Must be one of '
                             + format_dict_keys(HEAD_TYPE_VALUES) + '. If set to "' + AUTOMATIC + '", the most '
                             + 'suitable type is chosen automatically based on the parameter ' + PARAM_LOSS + '. For '
                             + 'additional options refer to the documentation.')
    parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT, type=str,
                        help='Whether potential refinements of rules should be searched for in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. If set to "' + AUTOMATIC
                             + '", the most suitable strategy is chosen automatically based on the parameter '
                             + PARAM_LOSS + '. For additional options refer to the documentation.')
    parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE, type=str,
                        help='Whether the gradients and Hessians for different examples should be calculated in '
                             + 'parallel or not. Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. If set '
                             + 'to "' + AUTOMATIC + '", the most suitable strategy is chosen automatically based on '
                             + 'the parameter ' + PARAM_LOSS + '. For additional options refer to the documentation.')


def main():
    parser = ArgumentParser(description='Allows to run experiments using the BOOMER algorithm')
    __add_arguments(parser)
    runnable = BoomerRunnable()
    runnable.run(parser)


if __name__ == '__main__':
    main()
