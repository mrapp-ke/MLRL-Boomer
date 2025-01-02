"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Integrates the BOOMER algorithm with the command line utility 'mlrl-testbed', which may be installed as an optional
dependency.
"""
from typing import Optional

from mlrl.boosting.boosting_learners import BoomerClassifier, BoomerRegressor
from mlrl.boosting.config import BOOMER_CLASSIFIER_PARAMETERS, BOOMER_REGRESSOR_PARAMETERS
from mlrl.boosting.cython.learner_boomer import BoomerClassifierConfig, BoomerRegressorConfig
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
                         classifier_type=BoomerClassifier,
                         classifier_config_type=BoomerClassifierConfig,
                         classifier_parameters=BOOMER_CLASSIFIER_PARAMETERS,
                         regressor_type=BoomerRegressor,
                         regressor_config_type=BoomerRegressorConfig,
                         regressor_parameters=BOOMER_REGRESSOR_PARAMETERS)

    def get_program_info(self) -> Optional[Runnable.ProgramInfo]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_program_info`
        """
        package_info = get_package_info()
        return Runnable.ProgramInfo(name='BOOMER',
                                    version=package_info.package_version,
                                    year='2020 - 2025',
                                    authors=['Michael Rapp et al.'],
                                    python_packages=[package_info])
