"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for dealing with different types of predictions.
"""
from enum import Enum

from mlrl.common.util.format import format_enum_values


class PredictionType(Enum):
    """
    Specifies all possible types of predictions that may be obtained from a machine learning model.
    """
    BINARY = 'binary'
    SCORES = 'scores'
    PROBABILITIES = 'probabilities'

    @staticmethod
    def parse(parameter_name: str, value: str) -> 'PredictionType':
        """
        Parses and returns a parameter value that specifies the `PredictionType` of the predictions to be obtained from
        a model. If the given value is invalid, a `ValueError` is raised.

        :param parameter_name:  The name of the parameter
        :param value:           The value to be parsed
        :return:                A `PredictionType`
        """
        try:
            return PredictionType(value)
        except ValueError as error:
            raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                             + format_enum_values(PredictionType) + ', but is "' + str(value) + '"') from error
