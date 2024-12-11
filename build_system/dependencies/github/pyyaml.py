"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for reading the contents of YAML files via "pyyaml".
"""
from functools import cached_property
from typing import Dict

from core.build_unit import BuildUnit
from util.io import TextFile, read_file
from util.pip import Pip


class YamlFile(TextFile):
    """
    A YAML file.
    """

    def __init__(self, build_unit: BuildUnit, file: str):
        """
        :param build_unit:  The build unit from which the YAML file is read
        :param file:        The path to the YAML file
        """
        super().__init__(file)
        self.build_unit = build_unit

    @cached_property
    def yaml_dict(self) -> Dict:
        """
        A dictionary that stores the content of the YAML file.
        """
        Pip.for_build_unit(self.build_unit).install_packages('pyyaml')
        # pylint: disable=import-outside-toplevel
        import yaml
        with read_file(self.file) as file:
            return yaml.load(file.read(), Loader=yaml.CLoader)

    def write_lines(self, *lines: str):
        super().write_lines(lines)
        del self.yaml_dict
