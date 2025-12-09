"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from typing import List, Optional, Type, override

from sklearn.base import BaseEstimator as SkLearnBaseEstimator, ClassifierMixin as SkLearnClassifierMixin

from ..common.cmd_builder import CmdBuilder
from ..common.cmd_builder_classification import ClassificationCmdBuilder
from ..common.datasets import Dataset


class SkLearnClassifierCmdBuilder(ClassificationCmdBuilder):
    """
    A builder that allows to configure a command for running a scikit-learn classifier.
    """

    def __init__(self, estimator_type: Type[SkLearnClassifierMixin], dataset: str = Dataset.EMOTIONS):
        super().__init__(expected_output_dir=CmdBuilder.EXPECTED_OUTPUT_DIR / 'sklearn' / 'classification',
                         input_dir=CmdBuilder.INPUT_DIR / 'sklearn',
                         batch_config=CmdBuilder.CONFIG_DIR / 'sklearn' / 'classification' / 'batch_config.yml',
                         runnable_module_name='mlrl.testbed_sklearn',
                         dataset=dataset)
        self._estimator_type = estimator_type
        self._meta_estimator_type: Optional[Type[SkLearnBaseEstimator]] = None

    def meta_estimator(self, meta_estimator_type: Optional[Type[SkLearnBaseEstimator]]):
        """
        Sets a meta-classifier to be used.

        :param meta_estimator_type: The type of the meta-classifier to be used or None, if no meta-classifier should be
                                    used
        :return:                    The builder itself
        """
        self._meta_estimator_type = meta_estimator_type
        return self

    def estimator(self, estimator_type: Type[SkLearnClassifierMixin]):
        """
        Sets a classifier to be used.

        :param estimator_type:  The type of the classifier to be used
        :return:                The builder itself
        """
        self._estimator_type = estimator_type
        return self

    @override
    def build(self) -> List[str]:
        args = super().build()
        meta_estimator_type = self._meta_estimator_type

        if meta_estimator_type is not None:
            args.append('--meta-estimator')
            args.append(meta_estimator_type.__name__)

        args.append('--estimator')
        args.append(self._estimator_type.__name__)

        return args
