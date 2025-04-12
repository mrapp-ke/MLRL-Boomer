"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing input or output data.
"""
from abc import ABC
from dataclasses import dataclass, replace
from typing import Type


class Data(ABC):
    """
    An abstract class for all classes that represent data that can be processed by connectors.
    """

    @dataclass
    class Context:
        """
        Specifies the aspects of an `ExperimentState` that should be taken into account for finding a suitable connector
        to exchange data with.

        Attributes:
            include_dataset_type:       True, if the type of the dataset should be taken into account, False otherwise
            include_prediction_scope:   True, if the scope of predictions should be taken into account, False otherwise
            include_fold:               True, if the cross validation fold should be taken into account, False otherwise
        """
        include_dataset_type: bool = True
        include_prediction_scope: bool = True
        include_fold: bool = True

    def __init__(self, default_context: Context = Context()):
        """
        :param default_context: A `Data.Context` to be used by default for finding a suitable connector this data can be
                                exchanged with
        """
        self.default_context = default_context
        self.custom_context = {}

    def get_context(self, connector_type: Type) -> Context:
        """
        Returns a `Data.Context` to can be used for finding a suitable connector of a specific type this data can be
        exchanged with.

        :param connector_type:  The type of the connector to search for
        :return:                A `Data.Context`
        """
        return self.custom_context.setdefault(connector_type, replace(self.default_context))
