"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "isort".
"""
from util.run import Program


class PyLint(Program):
    """
    Allows to run the external program "pylint".
    """

    def __init__(self, directory: str):
        """
        :param directory: The path to the directory, the program should be applied to
        """
        super().__init__('pylint', directory, '--jobs=0', '--recursive=y', '--ignore=build', '--rcfile=.pylintrc',
                         '--score=n')
