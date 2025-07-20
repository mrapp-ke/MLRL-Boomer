"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing TOML files via "toml".
"""
import tomllib

from functools import cached_property
from typing import Any, Dict, override

from core.build_unit import BuildUnit
from util.io import TextFile, read_file


class TomlFile(TextFile):
    """
    A TOML file.
    """

    def __init__(self, build_unit: BuildUnit, file: str):
        """
        :param build_unit:  The build unit from which the TOML file is read
        :param file:        The path to the TOML file
        """
        super().__init__(file)
        self.build_unit = build_unit

    @cached_property
    def toml_dict(self) -> Dict[Any, Any]:
        """
        A dictionary that stores the content of the TOML file.
        """
        with read_file(self.file) as file:
            toml_dict = tomllib.loads(file.read())
            return toml_dict if toml_dict else {}

    @override
    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.toml_dict
        except AttributeError:
            pass
