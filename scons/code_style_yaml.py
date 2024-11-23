"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from glob import glob
from os import path

from util.run import Program

YAML_DIRS = [('.', False), ('.github', True)]


class Yamlfix(Program):
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


def check_yaml_code_style(**_):
    """
    Checks if the YAML files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for directory, recursive in YAML_DIRS:
        print('Checking YAML files in the directory "' + directory + '"...')
        Yamlfix(directory, recursive=recursive).run()


def enforce_yaml_code_style(**_):
    """
    Enforces the YAML files to adhere to the code style definitions.
    """
    for directory, recursive in YAML_DIRS:
        print('Formatting YAML files in the directory "' + directory + '"...')
        Yamlfix(directory, recursive=recursive, enforce_changes=True).run()
