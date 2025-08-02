"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for measuring time.
"""
from dataclasses import dataclass
from functools import reduce
from timeit import default_timer as current_time
from typing import override


class Timer:
    """
    Allows measuring the execution time of code.
    """

    @dataclass
    class Duration:
        """
        Represents a specific duration.

        Attributes:
            value: The value (in fractional seconds) representing the duration
        """
        value: float = 0.0

        @override
        def __str__(self) -> str:
            seconds, millis = divmod(self.value, 1)
            millis = int(millis * 1000)
            days, seconds = divmod(seconds, 86400)
            days = int(days)
            hours, seconds = divmod(seconds, 3600)
            hours = int(hours)
            minutes, seconds = divmod(seconds, 60)
            minutes = int(minutes)
            seconds = int(seconds)
            substrings = []

            if days > 0:
                substrings.append(str(days) + ' day' + ('' if days == 1 else 's'))

            if hours > 0:
                substrings.append(str(hours) + ' hour' + ('' if hours == 1 else 's'))

            if minutes > 0:
                substrings.append(str(minutes) + ' minute' + ('' if minutes == 1 else 's'))

            if seconds > 0:
                substrings.append(str(seconds) + ' second' + ('' if seconds == 1 else 's'))

            if millis > 0 or len(substrings) == 0:
                substrings.append(str(millis) + ' millisecond' + ('' if millis == 1 else 's'))

            return reduce(
                lambda aggr, x: aggr + ((' and ' if x[0] == len(substrings) - 1 else ', ') if aggr else '') + x[1],
                enumerate(substrings), '')

    @dataclass
    class Time:
        """
        Represents a specific point in time.

        Attributes:
            time: The time in "float seconds"
        """
        time: float

        def __sub__(self, other: 'Timer.Time') -> 'Timer.Duration':
            return Timer.Duration(self.time - other.time)

    @staticmethod
    def start() -> Time:
        """
        Measures and returns the current time.

        :return: The current time
        """
        return Timer.Time(current_time())

    @staticmethod
    def stop(start_time: Time) -> Duration:
        """
        Measures and returns the execution time that has passed since a given `Time`.

        :param start_time:  A specific time in the past
        :return:            A `Duration` representing the time that has passed
        """
        stop_time = Timer.Time(current_time())
        return stop_time - start_time
