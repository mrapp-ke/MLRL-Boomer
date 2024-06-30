"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from mlrl.seco.config import SECO_RULE_LEARNER_PARAMETERS
from mlrl.seco.cython.learner_seco import SeCoClassifierConfig
from mlrl.seco.info import get_package_info
from mlrl.seco.seco_learners import SeCoClassifier

from mlrl.testbed.runnables import RuleLearnerRunnable


class SeCoRunnable(RuleLearnerRunnable):
    """
    A program that performs experiments using the Separate-and-Conquer (SeCo) algorithm.
    """

    def _get_program_info() -> Optional[RuleLearnerRunnable.ProgramInfo]:
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
    SeCoRunnable(description='Allows to run experiments using the Separate-and-Conquer algorithm',
                 learner_name='seco',
                 learner_type=SeCoClassifier,
                 config_type=SeCoClassifierConfig,
                 parameters=SECO_RULE_LEARNER_PARAMETERS).run()


if __name__ == '__main__':
    main()
