"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.boosting.boosting_learners import Boomer
from mlrl.boosting.config import BOOSTING_RULE_LEARNER_PARAMETERS
from mlrl.boosting.cython.learner_boomer import BoomerConfig
from mlrl.boosting.info import get_package_info

from mlrl.testbed.runnables import RuleLearnerRunnable


def create_program_info() -> RuleLearnerRunnable.ProgramInfo:
    """
    Creates and returns information about the program.

    :return: The information that has been created
    """
    package_info = get_package_info()
    return RuleLearnerRunnable.ProgramInfo(name='BOOMER',
                                           version=package_info.package_version,
                                           year='2020 - 2024',
                                           authors=['Michael Rapp et al.'],
                                           python_packages=[package_info])


def main():
    """
    The main function to be executed when the program starts.
    """
    RuleLearnerRunnable(description='Allows to run experiments using the BOOMER algorithm',
                        learner_name='boomer',
                        program_info=create_program_info(),
                        learner_type=Boomer,
                        config_type=BoomerConfig,
                        parameters=BOOSTING_RULE_LEARNER_PARAMETERS).run()


if __name__ == '__main__':
    main()
