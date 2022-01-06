"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


def assert_greater_or_equal(name: str, value, threshold):
    """
    Raises an exception if a given value is not greater or equal to a specific threshold.

    :param name:        The name of the parameter, the value corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value < threshold:
        raise AssertionError('Invalid value given for parameter "' + name + '": Must be greater or equal to ' + str(
            threshold) + ', but is ' + str(value))
