"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.config import configure_argument_parser, create_kwargs_from_parameters

from mlrl.testbed.runnables import RuleLearnerRunnable

from mlrl.boosting.cython.learner_boomer import BoomerConfig
from mlrl.boosting.config import BOOSTING_RULE_LEARNER_PARAMETERS
from mlrl.boosting.boosting_learners import Boomer


class BoomerRunnable(RuleLearnerRunnable):

    def __init__(self):
        super().__init__(description='Allows to run experiments using the BOOMER algorithm', learner_name='boomer')

    def _configure_arguments(self, parser: ArgumentParser):
        super()._configure_arguments(parser)
        configure_argument_parser(parser, BoomerConfig, BOOSTING_RULE_LEARNER_PARAMETERS)

    def _create_learner(self, args):
        return Boomer(**create_kwargs_from_parameters(args, BoomerConfig, BOOSTING_RULE_LEARNER_PARAMETERS))


def main():
    BoomerRunnable().run()


if __name__ == '__main__':
    main()
