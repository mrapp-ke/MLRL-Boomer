"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.seco.config import SECO_RULE_LEARNER_PARAMETERS
from mlrl.seco.cython.learner_seco import SeCoConfig
from mlrl.seco.info import get_package_info
from mlrl.seco.seco_learners import SeCo

from mlrl.testbed.runnables import RuleLearnerRunnable


def create_program_info() -> RuleLearnerRunnable.ProgramInfo:
    """
    Creates and returns information about the program.

    :return: The information that has been created
    """
    package_info = get_package_info()
    return RuleLearnerRunnable.ProgramInfo(name='Multi-label SeCo',
                                           version=package_info.package_version,
                                           year='2020 - 2024',
                                           authors=['Michael Rapp et al.'],
                                           python_packages=[package_info])


def main():
    """
    The main function to be executed when the program starts.
    """
    RuleLearnerRunnable(description='Allows to run experiments using the Separate-and-Conquer algorithm',
                        learner_name='seco',
                        program_info=create_program_info(),
                        learner_type=SeCo,
                        config_type=SeCoConfig,
                        parameters=SECO_RULE_LEARNER_PARAMETERS).run()


if __name__ == '__main__':
    main()
