"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for math operations.
"""


def divide_or_zero(numerator: float, denominator: float) -> float:
    """
    Returns the result of the floating point division `numerator / denominator` or 0, if a division by zero occurs.

    :param numerator:   The numerator
    :param denominator: The denominator
    :return:            The result of the division or 0, if a division by zero occurred
    """
    return numerator / denominator if denominator else 0
