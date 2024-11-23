"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from code_style.yaml.yamlfix import YamlFix

YAML_DIRS = [('.', False), ('.github', True)]


def check_yaml_code_style(**_):
    """
    Checks if the YAML files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for directory, recursive in YAML_DIRS:
        print('Checking YAML files in the directory "' + directory + '"...')
        YamlFix(directory, recursive=recursive).run()


def enforce_yaml_code_style(**_):
    """
    Enforces the YAML files to adhere to the code style definitions.
    """
    for directory, recursive in YAML_DIRS:
        print('Formatting YAML files in the directory "' + directory + '"...')
        YamlFix(directory, recursive=recursive, enforce_changes=True).run()
