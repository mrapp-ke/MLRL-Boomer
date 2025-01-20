"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "cmake".
"""
from util.run import Program


class Cmake(Program):
    """
    Allows to run the external program "cmake".
    """

    def __init__(self, *arguments: str):
        """
        :param arguments: Optional arguments to be passed to the program "cmake"
        """
        super().__init__('cmake', *arguments)
        self.install_program(False)
        self.print_arguments(True)
