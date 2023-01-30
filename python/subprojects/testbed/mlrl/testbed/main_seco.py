"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.format import format_dict_keys
from mlrl.seco.seco_learners import MultiLabelSeCoRuleLearner, HEURISTIC_VALUES
from mlrl.testbed.args import add_max_rules_argument, add_time_limit_argument, \
    add_sequential_post_optimization_argument, add_label_sampling_argument, add_instance_sampling_argument, \
    add_feature_sampling_argument, add_partition_sampling_argument, add_rule_pruning_argument, \
    add_rule_induction_argument, add_parallel_prediction_argument, add_parallel_statistic_update_argument, \
    add_parallel_rule_refinement_argument
from mlrl.testbed.args_seco import add_head_type_argument, add_lift_function_argument, PARAM_HEURISTIC, \
    PARAM_PRUNING_HEURISTIC
from mlrl.testbed.runnables import RuleLearnerRunnable


class SeCoRunnable(RuleLearnerRunnable):

    def __init__(self):
        super().__init__('Allows to run experiments using the Separate-and-Conquer algorithm')

    def _configure_arguments(self, parser: ArgumentParser):
        super()._configure_arguments(parser)
        add_max_rules_argument(parser)
        add_time_limit_argument(parser)
        add_sequential_post_optimization_argument(parser)
        add_label_sampling_argument(parser)
        add_instance_sampling_argument(parser)
        add_feature_sampling_argument(parser)
        add_partition_sampling_argument(parser)
        add_rule_pruning_argument(parser)
        add_rule_induction_argument(parser)
        add_parallel_prediction_argument(parser)
        add_parallel_rule_refinement_argument(parser)
        add_parallel_statistic_update_argument(parser)
        add_head_type_argument(parser)
        add_lift_function_argument(parser)
        parser.add_argument(PARAM_HEURISTIC,
                            type=str,
                            help='The name of the heuristic to be used for learning rules. Must be one of '
                            + format_dict_keys(HEURISTIC_VALUES) + '. For additional options refer to the '
                            + 'documentation.')
        parser.add_argument(PARAM_PRUNING_HEURISTIC,
                            type=str,
                            help='The name of the heuristic to be used for pruning individual rules. Must be one of '
                            + format_dict_keys(HEURISTIC_VALUES) + '. For additional options refer to the '
                            + 'documentation.')

    def _create_learner(self, args):
        return MultiLabelSeCoRuleLearner(random_state=args.random_state,
                                         feature_format=args.feature_format,
                                         label_format=args.label_format,
                                         prediction_format=args.prediction_format,
                                         rule_induction=args.rule_induction,
                                         max_rules=args.max_rules,
                                         time_limit=args.time_limit,
                                         sequential_post_optimization=args.sequential_post_optimization,
                                         heuristic=args.heuristic,
                                         pruning_heuristic=args.pruning_heuristic,
                                         rule_pruning=args.rule_pruning,
                                         label_sampling=args.label_sampling,
                                         instance_sampling=args.instance_sampling,
                                         feature_sampling=args.feature_sampling,
                                         head_type=args.head_type,
                                         lift_function=args.lift_function,
                                         parallel_rule_refinement=args.parallel_rule_refinement,
                                         parallel_statistic_update=args.parallel_statistic_update,
                                         parallel_prediction=args.parallel_prediction)

    def _get_learner_name(self) -> str:
        return "seco"


def main():
    runnable = SeCoRunnable()
    runnable.run()


if __name__ == '__main__':
    main()
