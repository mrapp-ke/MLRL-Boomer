"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "cpplint".
"""
from util.run import Program


class CppLint(Program):
    """
    Allows to run the external program "cpplint".
    """

    def __init__(self, directory: str):
        """
        :param directory: The path to the directory, the program should be applied to
        """
        super().__init__('cpplint', directory, '--quiet', '--recursive')
