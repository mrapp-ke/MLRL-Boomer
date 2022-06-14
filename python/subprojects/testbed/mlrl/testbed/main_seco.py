"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.rule_learners import PARALLEL_VALUES
from mlrl.common.strings import format_dict_keys, format_string_set
from mlrl.testbed.args import add_rule_learner_arguments, PARAM_HEAD_TYPE, PARAM_PARALLEL_RULE_REFINEMENT, \
    PARAM_PARALLEL_STATISTIC_UPDATE
from mlrl.testbed.runnables import RuleLearnerRunnable

from mlrl.seco.seco_learners import SeCoRuleLearner, HEAD_TYPE_VALUES, HEURISTIC_VALUES, LIFT_FUNCTION_VALUES, \
    HEAD_TYPE_PARTIAL

PARAM_HEURISTIC = '--heuristic'

PARAM_PRUNING_HEURISTIC = '--pruning-heuristic'

PARAM_LIFT_FUNCTION = '--lift-function'


class SeCoRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return SeCoRuleLearner(random_state=args.random_state,
                               feature_format=args.feature_format,
                               label_format=args.label_format,
                               predicted_label_format=args.predicted_label_format,
                               rule_induction=args.rule_induction,
                               max_rules=args.max_rules,
                               time_limit=args.time_limit,
                               heuristic=args.heuristic,
                               pruning_heuristic=args.pruning_heuristic,
                               pruning=args.pruning,
                               post_optimization=args.post_optimization,
                               label_sampling=args.label_sampling,
                               instance_sampling=args.instance_sampling,
                               feature_sampling=args.feature_sampling,
                               holdout=args.holdout,
                               head_type=args.head_type,
                               lift_function=args.lift_function,
                               parallel_rule_refinement=args.parallel_rule_refinement,
                               parallel_statistic_update=args.parallel_statistic_update,
                               parallel_prediction=args.parallel_prediction)

    def _get_learner_name(self) -> str:
        return "seco"


def __add_arguments(parser: ArgumentParser):
    add_rule_learner_arguments(parser)
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
    parser.add_argument(PARAM_PARALLEL_RULE_REFINEMENT, type=str,
                        help='Whether potential refinements of rules should be searched for in parallel or not. Must '
                             + 'be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional options refer to '
                             + 'the documentation.')
    parser.add_argument(PARAM_PARALLEL_STATISTIC_UPDATE, type=str,
                        help='Whether the confusion matrices for different examples should be calculated in parallel '
                             + 'or not. Must be one of ' + format_dict_keys(PARALLEL_VALUES) + '. For additional '
                             + 'options refer to the documentation')


def main():
    parser = ArgumentParser(description='Allows to run experiments using the Separate-and-Conquer algorithm')
    __add_arguments(parser)
    runnable = SeCoRunnable()
    runnable.run(parser)


if __name__ == '__main__':
    main()
