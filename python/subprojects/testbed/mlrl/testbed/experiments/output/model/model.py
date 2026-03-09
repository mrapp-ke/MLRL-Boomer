"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing models that are part of output data.
"""
from typing import Any, override

from mlrl.testbed.experiments.input.model.model import InputModel
from mlrl.testbed.experiments.output.data import ObjectOutputData

from mlrl.util.options import Options


class OutputModel(ObjectOutputData):
    """
    Represents a model.
    """

    def __init__(self, learner: Any):
        """
        :param learner: The learner that stores the model
        """
        super().__init__(properties=InputModel.PROPERTIES, context=InputModel.CONTEXT)
        self.learner = learner

    @override
    def to_object(self, options: Options, **kwargs) -> Any | None:
        """
        See :func:`mlrl.testbed.experiments.output.data.ObjectOutputData.to_object`
        """
        return self.learner
