"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to a Sphinx documentation.
"""
from abc import ABC, abstractmethod

from core.modules import Module


class ApidocModule(Module, ABC):
    """
    An abstract base class for all modules that contain code for which an API documentation can be generated.
    """

    class Filter(Module.Filter, ABC):
        """
        A filter that matches apidoc modules.
        """

    def __init__(self, output_directory: str):
        """
        :param output_directory: The path to the directory where the API documentation should be stored
        """
        self.output_directory = output_directory

    @abstractmethod
    def create_reference(self) -> str:
        """
        Must be implemented by subclasses in order to create a reference to API documentation.

        :return: The reference that has been created
        """
