"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "yamlfix".
"""
from glob import glob
from os import path

from util.run import Program


class YamlFix(Program):
    """
    Allows to run the external program "yamlfix".
    """

    def __init__(self, directory: str, recursive: bool = False, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param recursive:       True, if the program should be applied to subdirectories, False otherwise
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('yamlfix', '--config-file', '.yamlfix.toml')
        self.add_conditional_arguments(not enforce_changes, '--check')
        glob_path = path.join(directory, '**', '*') if recursive else path.join(directory, '*')
        glob_path_hidden = path.join(directory, '**', '.*') if recursive else path.join(directory, '.*')
        yaml_files = [
            file for file in glob(glob_path) + glob(glob_path_hidden)
            if path.basename(file).endswith('.yml') or path.basename(file).endswith('.yaml')
        ]
        self.add_arguments(yaml_files)
        self.print_arguments(True)
