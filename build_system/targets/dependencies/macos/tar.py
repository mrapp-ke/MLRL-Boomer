"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "tar".
"""
from util.run import Program


class TarExtract(Program):
    """
    Allows to run the external program "tar" for extracting a file.
    """

    def __init__(self, file_to_extract: str, into_directory: str):
        """
        :param file_to_extract: The path to the file to be extracted
        :param into_directory:  The path to the directory where the extracted files should be stored
        """
        super().__init__('tar', '--extract', '--file', file_to_extract, '--directory', into_directory)
        self.install_program(False)
        self.print_arguments(True)
