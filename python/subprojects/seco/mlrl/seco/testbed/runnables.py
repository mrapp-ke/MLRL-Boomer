"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Integrates the Separate-and-Conquer (SeCo) algorithm with the command line utility 'mlrl-testbed', which may be
installed as an optional dependency.
"""
from typing import Optional, override

from mlrl.common.testbed.program_info import RuleLearnerProgramInfo
from mlrl.common.testbed.runnables import RuleLearnerRunnable

from mlrl.seco.config.parameters import SECO_CLASSIFIER_PARAMETERS
from mlrl.seco.cython.learner_seco import SeCoClassifierConfig
from mlrl.seco.learners import SeCoClassifier
from mlrl.seco.package_info import get_package_info

from mlrl.testbed.program_info import ProgramInfo


class SeCoRunnable(RuleLearnerRunnable):
    """
    A program that allows performing experiments using the Separate-and-Conquer (SeCo) algorithm.
    """

    def __init__(self):
        super().__init__(classifier_type=SeCoClassifier,
                         classifier_config_type=SeCoClassifierConfig,
                         classifier_parameters=SECO_CLASSIFIER_PARAMETERS,
                         regressor_type=None,
                         regressor_config_type=None,
                         regressor_parameters=None)

    @override
    def get_program_info(self) -> Optional[ProgramInfo]:
        """
        See :func:`mlrl.testbed.runnables.Runnable.get_program_info`
        """
        package_info = get_package_info()
        return RuleLearnerProgramInfo(
            program_info=ProgramInfo(
                name='Multi-label SeCo',
                version=package_info.package_version,
                year='2020 - 2025',
                authors=['Michael Rapp et al.'],
            ),
            python_packages=[package_info],
        )
