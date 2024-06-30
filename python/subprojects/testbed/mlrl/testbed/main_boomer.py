"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import Optional

from python.subprojects.testbed.mlrl.testbed.runnables import Runnable

from mlrl.boosting.boosting_learners import BoomerClassifier
from mlrl.boosting.config import BOOSTING_RULE_LEARNER_PARAMETERS
from mlrl.boosting.cython.learner_boomer import BoomerClassifierConfig
from mlrl.boosting.info import get_package_info

from mlrl.testbed.runnables import RuleLearnerRunnable


class BoomerRunnable(RuleLearnerRunnable):
    """
    A program that performs experiments using the BOOMER algorithm.
    """

    def _get_program_info(self) -> Optional[Runnable.ProgramInfo]:
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
    BoomerRunnable(description='Allows to run experiments using the BOOMER algorithm',
                   learner_name='boomer',
                   learner_type=BoomerClassifier,
                   config_type=BoomerClassifierConfig,
                   parameters=BOOSTING_RULE_LEARNER_PARAMETERS).run()


if __name__ == '__main__':
    main()
