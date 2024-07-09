"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Integrates the BOOMER algorithm with the command line utility 'mlrl-testbed', which may be installed as an optional
dependency.
"""
from typing import Optional

from mlrl.boosting.boosting_learners import BoomerClassifier
from mlrl.boosting.config import BOOSTING_RULE_LEARNER_PARAMETERS
from mlrl.boosting.cython.learner_boomer import BoomerClassifierConfig
from mlrl.boosting.info import get_package_info

try:
    from mlrl.testbed.runnables import RuleLearnerRunnable, Runnable
except ImportError as error:
    raise ImportError('Optional dependency "mlrl-testbed" is not installed') from error


class BoomerRunnable(RuleLearnerRunnable):
    """
    A program that allows performing experiments using the BOOMER algorithm.
    """

    def __init__(self):
        super().__init__(learner_name='boomer',
                         learner_type=BoomerClassifier,
                         config_type=BoomerClassifierConfig,
                         parameters=BOOSTING_RULE_LEARNER_PARAMETERS)

    def get_program_info(self) -> Optional[Runnable.ProgramInfo]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_program_info`
        """
        package_info = get_package_info()
        return Runnable.ProgramInfo(name='BOOMER',
                                    version=package_info.package_version,
                                    year='2020 - 2024',
                                    authors=['Michael Rapp et al.'],
                                    python_packages=[package_info])
