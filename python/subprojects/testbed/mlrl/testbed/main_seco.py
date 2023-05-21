"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.seco.config import SECO_RULE_LEARNER_PARAMETERS
from mlrl.seco.cython.learner_seco import MultiLabelSeCoRuleLearnerConfig
from mlrl.seco.seco_learners import MultiLabelSeCoRuleLearner

from mlrl.testbed.runnables import RuleLearnerRunnable


def main():
    RuleLearnerRunnable(description='Allows to run experiments using the Separate-and-Conquer algorithm',
                        learner_name='seco',
                        learner_type=MultiLabelSeCoRuleLearner,
                        config_type=MultiLabelSeCoRuleLearnerConfig,
                        parameters=SECO_RULE_LEARNER_PARAMETERS).run()


if __name__ == '__main__':
    main()
