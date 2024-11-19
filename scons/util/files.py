"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for listing files and directories.
"""
from functools import partial, reduce
from glob import glob
from os import path
from typing import Callable, List


class DirectorySearch:
    """
    Allows to search for subdirectories.
    """

    Filter = Callable[[str, str], bool]

    def __init__(self):
        self.recursive = False
        self.excludes = []

    def set_recursive(self, recursive: bool) -> 'DirectorySearch':
        """
        Sets whether the search should be recursive or not.

        :param recursive:   True, if the search should be recursive, False otherwise
        :return:            The `DirectorySearch` itself
        """
        self.recursive = recursive
        return self

    def exclude(self, *excludes: Filter) -> 'DirectorySearch':
        """
        Sets one or several filters that should be used for excluding subdirectories.

        :param excludes:    The filters to be set
        :return:            The `DirectorySearch` itself
        """
        self.excludes.extend(excludes)
        return self

    def exclude_by_name(self, *names: str) -> 'DirectorySearch':
        """
        Sets one or several filters that should be used for excluding subdirectories by their names.

        :param names:   The names of the subdirectories to be excluded
        :return:        The `DirectorySearch` itself
        """

        def filter_directory(excluded_name: str, _: str, directory_name: str):
            return directory_name == excluded_name

        return self.exclude(*[partial(filter_directory, name) for name in names])

    def list(self, *directories: str) -> List[str]:
        """
        Lists all subdirectories that can be found in given directories.

        :param directories: The directories to search for subdirectories
        :return:            A list that contains all subdirectories that have been found
        """
        result = []

        def filter_file(file: str) -> bool:
            return path.isdir(file) and not reduce(
                lambda aggr, exclude: aggr or exclude(path.dirname(file), path.basename(file)), self.excludes, False)

        for directory in directories:
            subdirectories = [file for file in glob(path.join(directory, '*')) if filter_file(file)]

            if self.recursive:
                subdirectories.extend(self.list(*subdirectories))

            result.extend(subdirectories)

        return result
