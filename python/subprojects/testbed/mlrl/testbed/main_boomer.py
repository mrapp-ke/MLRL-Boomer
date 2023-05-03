"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.config import configure_argument_parser

from mlrl.testbed.runnables import RuleLearnerRunnable

from mlrl.boosting.cython.learner_boomer import BoomerConfig
from mlrl.boosting.config import BOOSTING_RULE_LEARNER_PARAMETERS
from mlrl.boosting.boosting_learners import Boomer


class BoomerRunnable(RuleLearnerRunnable):

    def __init__(self):
        super().__init__('Allows to run experiments using the BOOMER algorithm')

    def _configure_arguments(self, parser: ArgumentParser):
        super()._configure_arguments(parser)
        configure_argument_parser(parser, BoomerConfig, BOOSTING_RULE_LEARNER_PARAMETERS)

    def _create_learner(self, args):
        return Boomer(random_state=args.random_state,
                      feature_format=args.feature_format,
                      label_format=args.label_format,
                      prediction_format=args.prediction_format,
                      statistic_format=args.statistic_format,
                      default_rule=args.default_rule,
                      rule_induction=args.rule_induction,
                      max_rules=args.max_rules,
                      time_limit=args.time_limit,
                      global_pruning=args.global_pruning,
                      sequential_post_optimization=args.sequential_post_optimization,
                      loss=args.loss,
                      binary_predictor=args.binary_predictor,
                      probability_predictor=args.probability_predictor,
                      rule_pruning=args.rule_pruning,
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
        return 'boomer'


def main():
    runnable = BoomerRunnable()
    runnable.run()


if __name__ == '__main__':
    main()
