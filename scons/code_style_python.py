"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for checking and enforcing code style definitions.
"""
from modules import BUILD_MODULE, DOC_MODULE, PYTHON_MODULE
from util.run import Program

MD_DIRS = [('.', False), (DOC_MODULE.root_dir, True), (PYTHON_MODULE.root_dir, True)]

YAML_DIRS = [('.', False), ('.github', True)]


class Yapf(Program):
    """
    Allows to run the external program "yapf".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('yapf', '-r', '-p', '--style=.style.yapf', '--exclude', '**/build/*.py',
                         '-i' if enforce_changes else '--diff', directory)


class Isort(Program):
    """
    Allows to run the external program "isort".
    """

    def __init__(self, directory: str, enforce_changes: bool = False):
        """
        :param directory:       The path to the directory, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        super().__init__('isort', directory, '--settings-path', '.', '--virtual-env', 'venv', '--skip-gitignore')
        self.add_conditional_arguments(not enforce_changes, '--check')


class Pylint(Program):
    """
    Allows to run the external program "pylint".
    """

    def __init__(self, directory: str):
        """
        :param directory: The path to the directory, the program should be applied to
        """
        super().__init__('pylint', directory, '--jobs=0', '--recursive=y', '--ignore=build', '--rcfile=.pylintrc',
                         '--score=n')


def check_python_code_style(**_):
    """
    Checks if the Python source files adhere to the code style definitions. If this is not the case, an error is raised.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE]:
        directory = module.root_dir
        print('Checking Python code style in directory "' + directory + '"...')
        Isort(directory).run()
        Yapf(directory).run()
        Pylint(directory).run()


def enforce_python_code_style(**_):
    """
    Enforces the Python source files to adhere to the code style definitions.
    """
    for module in [BUILD_MODULE, PYTHON_MODULE, DOC_MODULE]:
        directory = module.root_dir
        print('Formatting Python code in directory "' + directory + '"...')
        Isort(directory, enforce_changes=True).run()
        Yapf(directory, enforce_changes=True).run()
