"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for listing files and directories.
"""
from functools import partial, reduce
from glob import glob
from os import path
from typing import Callable, List

from util.languages import Language


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


class FileSearch:
    """
    Allows to search for files.
    """

    def __init__(self):
        self.hidden = False
        self.file_patterns = {'*'}
        self.directory_search = DirectorySearch()

    def set_recursive(self, recursive: bool) -> 'FileSearch':
        """
        Sets whether the search should be recursive or not.

        :param recursive:   True, if the search should be recursive, False otherwise
        :return:            The `FileSearch` itself
        """
        self.directory_search.set_recursive(recursive)
        return self

    def exclude_subdirectories(self, *excludes: DirectorySearch.Filter) -> 'FileSearch':
        """
        Sets one or several filters that should be used for excluding subdirectories. Does only have an effect if the
        search is recursive.

        :param excludes:    The filters to be set
        :return:            The `FileSearch` itself
        """
        self.directory_search.exclude(*excludes)
        return self

    def exclude_subdirectories_by_name(self, *names: str) -> 'FileSearch':
        """
        Sets one or several filters that should be used for excluding subdirectories by their names. Does only have an
        effect if the search is recursive.

        :param names:   The names of the subdirectories to be excluded
        :return:        The `FileSearch` itself
        """
        self.directory_search.exclude_by_name(*names)
        return self

    def set_hidden(self, hidden: bool) -> 'FileSearch':
        """
        Sets whether hidden files should be included or not.

        :param hidden:  True, if hidden files should be included, False otherwise
        """
        self.hidden = hidden
        return self

    def set_suffixes(self, *suffixes: str) -> 'FileSearch':
        """
        Sets the suffixes of the files that should be included.

        :param suffixes:    The suffixes of the files that should be included (without starting dot)
        :return:            The `FileSearch` itself
        """
        self.file_patterns = {'*.' + suffix for suffix in suffixes}
        return self

    def set_languages(self, *languages: Language) -> 'FileSearch':
        """
        Sets the suffixes of the files that should be included.

        :param languages:   The languages of the files that should be included
        :return:            The `FileSearch` itself
        """
        return self.set_suffixes(*reduce(lambda aggr, language: aggr | language.value, languages, set()))

    def list(self, *directories: str) -> List[str]:
        """
        Lists all files that can be found in given directories.

        :param directories: The directories to search for files
        :return:            A list that contains all files that have been found
        """
        result = []
        subdirectories = self.directory_search.list(*directories) if self.directory_search.recursive else []

        for directory in list(directories) + subdirectories:
            for file_pattern in self.file_patterns:
                files = [file for file in glob(path.join(directory, file_pattern)) if path.isfile(file)]

                if self.hidden:
                    files.extend([file for file in glob(path.join(directory, '.' + file_pattern)) if path.isfile(file)])

                result.extend(files)

        return result
