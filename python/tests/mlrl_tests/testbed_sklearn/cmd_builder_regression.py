"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import override

from sklearn.base import BaseEstimator as SkLearnBaseEstimator, RegressorMixin as SkLearnRegressorMixin

from ..cmd_builder import CmdBuilder
from ..cmd_builder_regression import RegressionCmdBuilder
from ..datasets import Dataset


class SkLearnRegressorCmdBuilder(RegressionCmdBuilder):
    """
    A builder that allows to configure a command for running a scikit-learn regressor.
    """

    def __init__(self, estimator_type: type[SkLearnRegressorMixin], dataset: str = Dataset.ATP7D):
        super().__init__(expected_output_dir=CmdBuilder.EXPECTED_OUTPUT_DIR / 'sklearn' / 'regression',
                         input_dir=CmdBuilder.INPUT_DIR / 'sklearn',
                         batch_config=CmdBuilder.CONFIG_DIR / 'sklearn' / 'regression' / 'batch_config.yml',
                         runnable_module_name='mlrl.testbed_sklearn',
                         dataset=dataset)
        self._estimator_type = estimator_type
        self._meta_estimator_type: type[SkLearnBaseEstimator] | None = None

    def meta_estimator(self, meta_estimator_type: type[SkLearnBaseEstimator] | None):
        """
        Sets a meta-regressor to be used.

        :param meta_estimator_type: The type of the meta-regressor to be used or None, if no meta-regressor should be
                                    used
        :return:                    The builder itself
        """
        self._meta_estimator_type = meta_estimator_type
        return self

    def estimator(self, estimator_type: type[SkLearnRegressorMixin]):
        """
        Sets a regressor to be used.

        :param estimator_type:  The type of the regressor to be used
        :return:                The builder itself
        """
        self._estimator_type = estimator_type
        return self

    @override
    def build(self) -> list[str]:
        args = super().build()
        meta_estimator_type = self._meta_estimator_type

        if meta_estimator_type is not None:
            args.append('--meta-estimator')
            args.append(meta_estimator_type.__name__)

        args.append('--estimator')
        args.append(self._estimator_type.__name__)

        return args
