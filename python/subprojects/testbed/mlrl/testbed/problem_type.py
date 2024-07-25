"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility classes for dealing with different types of machine learning problems.
"""
from enum import Enum

from mlrl.common.format import format_enum_values


class ProblemType(Enum):
    """
    All supported type of machine learning problems.
    """
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

    @staticmethod
    def parse(parameter_name: str, value: str) -> 'ProblemType':
        """
        Parses and returns a parameter value that specifies the `ProblemType` of a machine learning problem to be
        solved. If the given value is invalid, a `ValueError` is raised.

        :param parameter_name:  The name of the parameter
        :param value:           The value to be parsed
        :return:                A `ProblemType`
        """
        try:
            return ProblemType(value)
        except ValueError as error:
            raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                             + format_enum_values(ProblemType) + ', but is "' + str(value) + '"') from error
