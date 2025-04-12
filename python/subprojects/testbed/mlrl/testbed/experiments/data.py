"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input or output data.
"""
from abc import ABC
from dataclasses import replace
from typing import Type

from mlrl.testbed.experiments.state import ExperimentState


class Data(ABC):
    """
    An abstract class for all classes that represent data that can be processed by connectors.
    """

    def __init__(self, default_context: ExperimentState.Context = ExperimentState.Context()):
        """
        :param default_context: An `ExperimentState.Context` to be used by default for finding a suitable connector this
                               output data can be processed by
        """
        self.default_context = default_context
        self.custom_context = {}

    def get_context(self, connector_type: Type) -> ExperimentState.Context:
        """
        Returns an `ExperimentState.Context` to can be used for finding a suitable connector of a specific type this
        data can be written too.

        :param connector_type:  The type of the connector to search for
        :return:                An `ExperimentState.Context`
        """
        return self.custom_context.setdefault(connector_type, replace(self.default_context))
