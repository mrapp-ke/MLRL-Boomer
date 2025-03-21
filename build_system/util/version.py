"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for dealing with version numbers.
"""
from dataclasses import dataclass
from functools import reduce
from typing import Tuple


@dataclass
class Version:
    """
    Represents a version.

    Attributes:
        numbers: The version numbers
    """
    numbers: Tuple[int, ...]

    @staticmethod
    def parse_version_number(version_number: str) -> int:
        """
        Parses and returns a single version number from a given string.

        :param version_number:  The string to be parsed
        :return:                The version number that has been parsed
        """
        try:
            number = int(version_number)

            if number < 0:
                raise ValueError()

            return number
        except ValueError as error:
            raise ValueError('Version numbers must be non-negative integers, but got: ' + version_number) from error

    @staticmethod
    def parse(version: str, skip_on_error: bool = False) -> 'Version':
        """
        Parses and returns a version from a given string.

        :param version:         The string to be parsed
        :param skip_on_error:   True, if all remaining version numbers should be skipped if one of them could not be
                                parsed, False if an error should be raised
        :return:                The version that has been parsed
        """
        numbers = []

        for part in version.split('.'):
            try:
                numbers.append(Version.parse_version_number(part))
            except ValueError as error:
                if not skip_on_error:
                    raise error

        return Version(tuple(numbers))

    def __str__(self) -> str:
        return reduce(lambda aggr, number: aggr + ('.' if aggr else '') + str(number), self.numbers, '')

    def __eq__(self, other: 'Version') -> bool:
        return self.numbers == other.numbers

    def __ne__(self, other: 'Version') -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: 'Version'):
        first_numbers = self.numbers
        second_numbers = other.numbers
        limit = min(len(first_numbers), len(second_numbers))

        for i in range(limit):
            first_number = first_numbers[i]
            second_number = second_numbers[i]

            if first_number < second_number:
                return True

        return False

    def __le__(self, other: 'Version') -> bool:
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other: 'Version') -> bool:
        return not self.__le__(other)

    def __ge__(self, other: 'Version') -> bool:
        return not self.__lt__(other)
