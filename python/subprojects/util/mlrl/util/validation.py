"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for validation.
"""
from typing import override


class ValidationError(Exception):
    """
    An exception to be raised when a validation error occurs.
    """

    def __init__(self, message: str):
        """
        :param message: The message of the exception
        """
        super().__init__(message)
        self.message = message

    @override
    def __str__(self) -> str:
        return self.message


def assert_greater(name: str, value, threshold):
    """
    Raises a `ValidationError` if a given value is not greater than a specific threshold.

    :param name:        The name of the argument or parameter, the value  corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value <= threshold:
        raise ValidationError('Invalid value given for ' + ('argument' if name.startswith('-') else 'parameter') + ' "'
                              + name + '": Must be greater than ' + str(threshold) + ', but is ' + str(value))


def assert_greater_or_equal(name: str, value, threshold):
    """
    Raises a `ValidationError` if a given value is not greater or equal to a specific threshold.

    :param name:        The name of the argument or parameter, the value corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value < threshold:
        raise ValidationError('Invalid value given for ' + ('argument' if name.startswith('-') else 'parameter') + ' "'
                              + name + '": Must be greater or equal to ' + str(threshold) + ', but is ' + str(value))


def assert_less(name: str, value, threshold):
    """
    Raises a `ValidationError` if a given value is not less than a specific threshold.
    
    :param name:        The name of the argument or parameter, the value corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value >= threshold:
        raise ValidationError('Invalid value given for ' + ('argument' if name.startswith('-') else 'parameter') + ' "'
                              + name + '": Must be less than ' + str(threshold) + ', but is ' + str(value))


def assert_less_or_equal(name: str, value, threshold):
    """
    Raises a `ValidationError` if a given value is not less or equal to a specific threshold.

    :param name:        The name of the argument or parameter, the value corresponds to
    :param value:       The value
    :param threshold:   The threshold
    """
    if value > threshold:
        raise ValidationError('Invalid value given for ' + ('argument' if name.startswith('-') else 'parameter') + ' "'
                              + name + '": Must be less or equal to ' + str(threshold) + ', but is ' + str(value))


def assert_multiple(name: str, value, other):
    """
    Raises a `ValidationError` if a given value is not a multiple of another value.

    :param name:    The name of the argument or parameter, the value corresponds to
    :param value:   The value that should be a multiple of `other`
    :param other:   The other value
    """
    if value % other != 0:
        raise ValidationError('Invalid value given for ' + ('argument' if name.startswith('-') else 'parameter') + ' "'
                              + name + '": Must be a multiple of ' + str(other) + ', but is ' + str(value))


def assert_not_none(name: str, value):
    """
    Raises a `ValueError` if a given value is None.

    :param name:    The name of the argument or parameter, the value corresponds to
    :param value:   The value
    """
    if value is None:
        raise ValidationError('Invalid value given for ' + ('argument' if name.startswith('-') else 'parameter') + ' "'
                              + name + '": Must not be None')
