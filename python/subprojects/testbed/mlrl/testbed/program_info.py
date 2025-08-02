"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for providing the text to be shown when the "--version" flag is passed to the command line API.
"""
from dataclasses import dataclass, field
from typing import List, Optional, override

from mlrl.util.format import format_iterable


@dataclass
class ProgramInfo:
    """
    Provides information about a program.

    :param name:    The name of the program
    :param version: The version of the program
    :param year:    The year when the program was released
    :param authors: A list containing the names of all authors of the program
    """
    name: str
    version: str
    year: Optional[str] = None
    authors: List[str] = field(default_factory=list)

    def __get_copyright_text(self) -> str:
        text = ''
        year = self.year

        if year:
            text += ' ' + year

        authors = self.authors

        if authors:
            text += ' ' + format_iterable(authors)

        return 'Copyright (c)' + text if text else ''

    @override
    def __str__(self) -> str:
        program_info = self.name + ' ' + self.version
        copyright_text = self.__get_copyright_text()
        return program_info + '\n\n' + copyright_text if copyright_text else program_info
