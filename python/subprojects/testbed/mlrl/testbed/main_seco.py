"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.strings import format_dict_keys, format_string_set
from mlrl.testbed.args import add_rule_learner_arguments, add_max_rules_argument, add_time_limit_argument, \
    add_label_sampling_argument, add_instance_sampling_argument, add_feature_sampling_argument, \
    add_partition_sampling_argument, add_feature_binning_argument, add_pruning_argument, add_rule_induction_argument, \
    add_parallel_prediction_argument, add_parallel_statistic_update_argument, add_parallel_rule_refinement_argument, \
    PARAM_HEAD_TYPE
from mlrl.testbed.runnables import RuleLearnerRunnable

from mlrl.seco.seco_learners import MultiLabelSeCoRuleLearner, HEAD_TYPE_VALUES, HEURISTIC_VALUES, \
    LIFT_FUNCTION_VALUES, HEAD_TYPE_PARTIAL

PARAM_HEURISTIC = '--heuristic'

PARAM_PRUNING_HEURISTIC = '--pruning-heuristic'

PARAM_LIFT_FUNCTION = '--lift-function'


class SeCoRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return MultiLabelSeCoRuleLearner(random_state=args.random_state,
                                         feature_format=args.feature_format,
                                         label_format=args.label_format,
                                         predicted_label_format=args.predicted_label_format,
                                         rule_induction=args.rule_induction,
                                         max_rules=args.max_rules,
                                         time_limit=args.time_limit,
                                         heuristic=args.heuristic,
                                         pruning_heuristic=args.pruning_heuristic,
                                         pruning=args.pruning,
                                         label_sampling=args.label_sampling,
                                         instance_sampling=args.instance_sampling,
                                         feature_sampling=args.feature_sampling,
                                         holdout=args.holdout,
                                         feature_binning=args.feature_binning,
                                         head_type=args.head_type,
                                         lift_function=args.lift_function,
                                         parallel_rule_refinement=args.parallel_rule_refinement,
                                         parallel_statistic_update=args.parallel_statistic_update,
                                         parallel_prediction=args.parallel_prediction)

    def _get_learner_name(self) -> str:
        return "seco"


def __add_arguments(parser: ArgumentParser):
    add_rule_learner_arguments(parser)
    add_max_rules_argument(parser)
    add_time_limit_argument(parser)
    add_label_sampling_argument(parser)
    add_instance_sampling_argument(parser)
    add_feature_sampling_argument(parser)
    add_partition_sampling_argument(parser)
    add_feature_binning_argument(parser)
    add_pruning_argument(parser)
    add_rule_induction_argument(parser)
    add_parallel_prediction_argument(parser)
    add_parallel_rule_refinement_argument(parser)
    add_parallel_statistic_update_argument(parser)
    parser.add_argument(PARAM_HEURISTIC, type=str,
                        help='The name of the heuristic to be used for learning rules. Must be one of '
                             + format_dict_keys(HEURISTIC_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_PRUNING_HEURISTIC, type=str,
                        help='The name of the heuristic to be used for pruning rules. Must be one of '
                             + format_dict_keys(HEURISTIC_VALUES) + '. For additional options refer to the '
                             + 'documentation.')
    parser.add_argument(PARAM_LIFT_FUNCTION, type=str,
                        help='The lift function to be used for the induction of multi-label rules. Must be one of '
                             + format_dict_keys(LIFT_FUNCTION_VALUES) + '. Does only have an effect if the parameter '
                             + PARAM_HEAD_TYPE + ' is set to "' + HEAD_TYPE_PARTIAL + '".')
    parser.add_argument(PARAM_HEAD_TYPE, type=str,
                        help='The type of the rule heads that should be used. Must be one of '
                             + format_string_set(HEAD_TYPE_VALUES) + '.')


def main():
    parser = ArgumentParser(description='Allows to run experiments using the Separate-and-Conquer algorithm')
    __add_arguments(parser)
    runnable = SeCoRunnable()
    runnable.run(parser)


if __name__ == '__main__':
    main()
