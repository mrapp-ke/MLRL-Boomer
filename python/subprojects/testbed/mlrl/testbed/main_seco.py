"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.seco.config import SECO_RULE_LEARNER_PARAMETERS
from mlrl.seco.cython.learner_seco import MultiLabelSeCoRuleLearnerConfig
from mlrl.seco.info import get_package_info
from mlrl.seco.seco_learners import MultiLabelSeCoRuleLearner

from mlrl.testbed.runnables import RuleLearnerRunnable


def create_program_info() -> RuleLearnerRunnable.ProgramInfo:
    package_info = get_package_info()
    return RuleLearnerRunnable.ProgramInfo(name='Multi-label SeCo',
                                           version=package_info.get_package_version(),
                                           year='2020 - 2023',
                                           authors=['Michael Rapp et al.'])


def main():
    RuleLearnerRunnable(description='Allows to run experiments using the Separate-and-Conquer algorithm',
                        learner_name='seco',
                        program_info=create_program_info(),
                        learner_type=MultiLabelSeCoRuleLearner,
                        config_type=MultiLabelSeCoRuleLearnerConfig,
                        parameters=SECO_RULE_LEARNER_PARAMETERS).run()


if __name__ == '__main__':
    main()
