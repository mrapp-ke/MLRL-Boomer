"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing data.
"""
from dataclasses import dataclass


@dataclass
class Properties:
    """
    The properties of input or output data.

    Attributes:
        name:       A name to be included in log messages
        file_name:  A file name to be used when reading from or writing to files
    """
    name: str
    file_name: str
