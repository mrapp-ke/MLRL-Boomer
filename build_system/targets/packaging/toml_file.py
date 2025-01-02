"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for reading and writing TOML files via "toml".
"""
from functools import cached_property
from typing import Dict

from core.build_unit import BuildUnit
from util.io import TextFile, read_file
from util.pip import Pip


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
    def toml_dict(self) -> Dict:
        """
        A dictionary that stores the content of the TOML file.
        """
        Pip.for_build_unit(self.build_unit).install_packages('toml')
        # pylint: disable=import-outside-toplevel
        import toml
        with read_file(self.file) as file:
            toml_dict = toml.loads(file.read())
            return toml_dict if toml_dict else {}

    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.toml_dict
        except AttributeError:
            pass
