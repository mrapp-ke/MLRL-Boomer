"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing models that are part of output data.
"""
from typing import Any, Optional, override

from mlrl.testbed.experiments.context import Context
from mlrl.testbed.experiments.output.data import ObjectOutputData, OutputData

from mlrl.util.options import Options


class OutputModel(ObjectOutputData):
    """
    Represents a model.
    """

    def __init__(self, learner: Any):
        """
        :param learner: The learner that stores the model
        """
        super().__init__(OutputData.Properties(name='Model', file_name='model'),
                         Context(include_dataset_type=False, include_prediction_scope=False))
        self.learner = learner

    @override
    def to_object(self, options: Options, **kwargs) -> Optional[Any]:
        """
        See :func:`mlrl.testbed.experiments.output.data.ObjectOutputData.to_object`
        """
        return self.learner
