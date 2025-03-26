"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for converting output data into different representations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from mlrl.common.config.options import Options


class TextConverter(ABC):
    """
    An abstract base class for all classes that support conversion into a textual representation.
    """

    @abstractmethod
    def to_text(self, options: Options, **kwargs) -> Optional[str]:
        """
        Creates and returns a textual representation of the object.

        :param options: Options to be taken into account
        :return:        The textual representation that has been created
        """


class TableConverter(ABC):
    """
    An abstract base class for all classes that support conversion into a tabular representation.
    """

    Table = List[Dict[str, str]]

    @abstractmethod
    def to_table(self, options: Options, **kwargs) -> Optional[Table]:
        """
        Creates and returns a tabular representation of the object.

        :param options: Options to be taken into account
        :return:        The tabular representation that has been created
        """
