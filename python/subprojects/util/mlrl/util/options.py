"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides a data structure that allows to store and parse options that are provided as key-value pairs.
"""
from enum import Enum, EnumType
from typing import Any, Dict, Optional, Set, Tuple

from mlrl.util.format import format_enum_values, format_set


class BooleanOption(Enum):
    """
    Specifies all valid textual representations of boolean values.
    """
    TRUE = 'true'
    FALSE = 'false'

    @staticmethod
    def parse(text: str) -> bool:
        """
        Parses a given text that represents a boolean value. If the given text does not represent a valid boolean value,
        a `ValueError` is raised.

        :param text:    The text to be parsed
        :return:        True or false, depending on the given text
        """
        if text == BooleanOption.TRUE.value:
            return True
        if text == BooleanOption.FALSE.value:
            return False
        raise ValueError('Invalid boolean value given. Must be one of ' + format_enum_values(BooleanOption)
                         + ', but is "' + str(text) + '".')


class Options:
    """
    Stores key-value pairs in a dictionary and provides methods to access and validate them.
    """

    ERROR_MESSAGE_INVALID_SYNTAX = 'Invalid syntax used to specify additional options'

    ERROR_MESSAGE_INVALID_OPTION = 'Expected comma-separated list of key-value pairs'

    def __init__(self, dictionary: Optional[Dict[str, Any]] = None):
        self.dictionary = dictionary if dictionary is not None else {}

    @classmethod
    def create(cls, string: str, allowed_keys: Set[str]):
        """
        Parses the options that are provided via a given string that is formatted according to the following syntax:
        "[key1=value1,key2=value2]". If the given string is malformed, a `ValueError` will be raised.

        :param string:          The string to be parsed
        :param allowed_keys:    A set that contains all valid keys
        :return:                An object of type `Options` that stores the key-value pairs that have been parsed from
                                the given string
        """
        options = cls()

        if string:
            if not string.startswith('{'):
                raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. Must start with "{", but is "' + string
                                 + '"')
            if not string.endswith('}'):
                raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. Must end with "}", but is "' + string + '"')

            string = string[1:-1]

            if string:
                for option_index, option in enumerate(string.split(',')):
                    if option:
                        parts = option.split('=')

                        if len(parts) != 2:
                            raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. '
                                             + Options.ERROR_MESSAGE_INVALID_OPTION + ', but got element "' + option
                                             + '" at index ' + str(option_index))

                        key = parts[0]

                        if len(key) == 0:
                            raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. '
                                             + Options.ERROR_MESSAGE_INVALID_OPTION
                                             + ', but key is missing from element "' + option + '" at index '
                                             + str(option_index))

                        if key not in allowed_keys:
                            raise ValueError('Key must be one of ' + format_set(allowed_keys) + ', but got key "' + key
                                             + '" at index ' + str(option_index))

                        value = parts[1]

                        if len(value) == 0:
                            raise ValueError(Options.ERROR_MESSAGE_INVALID_SYNTAX + '. '
                                             + Options.ERROR_MESSAGE_INVALID_OPTION
                                             + ', but value is missing from element "' + option + '" at index '
                                             + str(option_index))

                        options.dictionary[key] = value

        return options

    def get_string(self, key: str, default_value: Optional[str] = None) -> Optional[str]:
        """
        Returns a string that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dictionary:
            return str(self.dictionary[key])

        return default_value

    def get_bool(self, key: str, default_value: bool) -> bool:
        """
        Returns a boolean that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dictionary:
            value = str(self.dictionary[key])
            return BooleanOption.parse(value)

        return default_value

    def get_int(self, key: str, default_value: int) -> int:
        """
        Returns an integer that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dictionary:
            value = self.dictionary[key]

            try:
                value = int(value)
            except ValueError as error:
                raise ValueError('Value for key "' + key + '" is expected to be an integer, but is "' + str(value)
                                 + '"') from error

            return value

        return default_value

    def get_float(self, key: str, default_value: float) -> float:
        """
        Returns a float that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dictionary:
            value = self.dictionary[key]

            try:
                value = float(value)
            except ValueError as error:
                raise ValueError('Value for key "' + key + '" is expected to be a float, but is "' + str(value)
                                 + '"') from error

            return value

        return default_value


def parse_enum(parameter_name: str,
               value: Optional[str],
               enum: EnumType,
               default: Optional[Enum] = None) -> Optional[EnumType]:
    """
    Parses and returns an enum value. If the given value is invalid, a `ValueError` is raised.

    :param parameter_name:  The name of the parameter
    :param value:           The value to be parsed
    :param enum:            The enum
    :param default:         The default value to be returned if `value` is None
    :return:                The value that has been parsed or `default`, if the given value is None
    """
    if value:
        for enum_value in enum:
            expected_value = enum_value.value if isinstance(enum_value.value, str) else enum_value.name.lower()

            if expected_value == value:
                return enum_value

        raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                         + format_enum_values(enum) + ', but is "' + str(value) + '"')

    return default


def parse_param(parameter_name: str, value: str, allowed_values: Set[str]) -> str:
    """
    Parses and returns a parameter value. If the given value is invalid, a `ValueError` is raised.

    :param parameter_name:  The name of the parameter
    :param value:           The value to be parsed
    :param allowed_values:  A set that contains all valid values
    :return:                The value that has been parsed
    """
    if value in allowed_values:
        return value

    raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                     + format_set(allowed_values) + ', but is "' + str(value) + '"')


def parse_param_and_options(parameter_name: str, value: str,
                            allowed_values_and_options: Dict[str, Set[str]]) -> Tuple[str, Options]:
    """
    Parses and returns a parameter value, as well as additional `Options` that may be associated with it. If the given
    value is invalid, a `ValueError` is raised.
    
    :param parameter_name:              The name of the parameter
    :param value:                       The value to be parsed
    :param allowed_values_and_options:  A dictionary that contains all valid values, as well as their options
    :return:                            A tuple that contains the value that has been parsed, as well as additional
                                        `Options`.
    """
    for allowed_value, allowed_options in allowed_values_and_options.items():
        if value.startswith(allowed_value):
            suffix = value[len(allowed_value):].strip()

            if suffix:
                try:
                    return allowed_value, Options.create(suffix, allowed_options)
                except ValueError as error:
                    raise ValueError('Invalid options specified for parameter "' + parameter_name + '" with value "'
                                     + allowed_value + '": ' + str(error)) from error

            return allowed_value, Options()

    raise ValueError('Invalid value given for parameter "' + parameter_name + '": Must be one of '
                     + format_set(allowed_values_and_options.keys()) + ', but is "' + value + '"')
