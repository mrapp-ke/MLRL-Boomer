"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Integrates the BOOMER algorithm with the command line utility 'mlrl-testbed', which may be installed as an optional
dependency.
"""
from typing import Optional, Set, override

from mlrl.common.testbed.program_info import RuleLearnerProgramInfo
from mlrl.common.testbed.runnables import RuleLearnerRunnable

from mlrl.boosting.config.parameters import BOOMER_CLASSIFIER_PARAMETERS, BOOMER_REGRESSOR_PARAMETERS
from mlrl.boosting.cython.learner_boomer import BoomerClassifierConfig, BoomerRegressorConfig
from mlrl.boosting.learners import BoomerClassifier, BoomerRegressor
from mlrl.boosting.package_info import get_package_info
from mlrl.boosting.testbed.experiments.output.probability_calibration.extension import \
    JointProbabilityCalibrationModelExtension, MarginalProbabilityCalibrationModelExtension

from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.program_info import ProgramInfo


class BoomerRunnable(RuleLearnerRunnable):
    """
    A program that allows performing experiments using the BOOMER algorithm.
    """

    def __init__(self):
        super().__init__(classifier_type=BoomerClassifier,
                         classifier_config_type=BoomerClassifierConfig,
                         classifier_parameters=BOOMER_CLASSIFIER_PARAMETERS,
                         regressor_type=BoomerRegressor,
                         regressor_config_type=BoomerRegressorConfig,
                         regressor_parameters=BOOMER_REGRESSOR_PARAMETERS)

    @override
    def get_extensions(self) -> Set[Extension]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_extensions`
        """
        return super().get_extensions() | {
            MarginalProbabilityCalibrationModelExtension(),
            JointProbabilityCalibrationModelExtension(),
        }

    @override
    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_program_info`
        """
        package_info = get_package_info()
        return RuleLearnerProgramInfo(
            program_info=ProgramInfo(
                name='BOOMER',
                version=package_info.package_version,
                year='2020 - 2025',
                authors=['Michael Rapp et al.'],
            ),
            python_packages=[package_info],
        )
