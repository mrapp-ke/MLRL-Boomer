"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.config import configure_argument_parser

from mlrl.seco.cython.learner_seco import MultiLabelSeCoRuleLearnerConfig
from mlrl.seco.config import SECO_RULE_LEARNER_PARAMETERS
from mlrl.seco.seco_learners import MultiLabelSeCoRuleLearner

from mlrl.testbed.runnables import RuleLearnerRunnable


class SeCoRunnable(RuleLearnerRunnable):

    def __init__(self):
        super().__init__(description='Allows to run experiments using the Separate-and-Conquer algorithm',
                         learner_name='seco')

    def _configure_arguments(self, parser: ArgumentParser):
        super()._configure_arguments(parser)
        configure_argument_parser(parser, MultiLabelSeCoRuleLearnerConfig, SECO_RULE_LEARNER_PARAMETERS)

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


def main():
    runnable = SeCoRunnable()
    runnable.run()


if __name__ == '__main__':
    main()
