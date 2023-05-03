"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from argparse import ArgumentParser

from mlrl.common.config import configure_argument_parser, create_kwargs_from_parameters

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
        return MultiLabelSeCoRuleLearner(**create_kwargs_from_parameters(args, SECO_RULE_LEARNER_PARAMETERS))


def main():
    SeCoRunnable().run()


if __name__ == '__main__':
    main()
