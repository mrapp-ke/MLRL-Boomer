#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a data structure that allows to store and parse options that are provided as key-value pairs.
"""


class Options:
    """
    Stores key-value pairs in a dictionary and provides methods to access and validate them.
    """

    def __init__(self):
        self.dict = {}

    @classmethod
    def create(cls, string: str):
        """
        Parses the options that are provided via a given string that is formatted according to the following syntax:
        "[key1=value1,key2=value2]". If the given string is malformed, a `ValueError` will be raised.

        :param string:  The string to be parsed
        :return:        An object of type `Options` that stores the key-value pairs that have been parsed from the given
                        string
        """
        options = cls()

        if string is not None and len(string) > 0:
            if not string.startswith('{') or not string.endswith('}'):
                raise ValueError('Invalid syntax used to specify key-value pairs: ' + string)

            string = string[1:-1]

            if len(string) > 0:
                for argument in string.split(','):
                    if len(argument) == 0:
                        raise ValueError('Invalid syntax used to specify key-value pairs: ' + string)

                    parts = argument.split('=')

                    if len(parts) != 2:
                        raise ValueError('Invalid syntax used to specify key-value pairs: ' + string)

                    key = parts[0]
                    value = parts[1]

                    if len(key) == 0 or len(value) == 0:
                        raise ValueError('Invalid syntax used to specify key-value pairs: ' + string)

                    options.dict[key] = value

        return options

    def get_string(self, key: str, default_value: str) -> str:
        """
        Returns a string that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dict:
            return str(self.dict[key])

        return default_value

    def get_bool(self, key: str, default_value: bool) -> bool:
        """
        Returns a boolean that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dict:
            value = str(self.dict[key]).lower()

            if value == 'false':
                return False
            if value == 'true':
                return True
            raise ValueError('Value for key \'' + key + '\' cannot be converted to boolean')

        return default_value

    def get_int(self, key: str, default_value: int) -> int:
        """
        Returns an integer that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                The value that is associated with the given key or the given default value
        """
        if key in self.dict:
            value = self.dict[key]

            try:
                value = int(value)
            except ValueError:
                raise ValueError('Value for key \'' + key + '\' cannot be converted to integer: ' + str(value))

            return value

        return default_value

    def get_float(self, key: str, default_value: float) -> float:
        """
        Returns a float that corresponds to a specific key.

        :param key:             The key
        :param default_value:   The default value to be returned, if no value is associated with the given key
        :return:                THe value that is associated with the given key or the given default value
        """
        if key in self.dict:
            value = self.dict[key]

            try:
                value = float(value)
            except ValueError:
                raise ValueError('Value for key \'' + key + '\' cannot be converted to float: ' + str(value))

            return value

        return default_value
