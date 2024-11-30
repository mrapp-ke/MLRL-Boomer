"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for listing files and directories.
"""
from functools import partial, reduce
from glob import glob
from os import path
from typing import Callable, List, Optional, Set

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

    Filter = Callable[[str, str], bool]

    def __init__(self):
        self.hidden = False
        self.filters = []
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

    def add_filters(self, *filter_functions: Filter) -> 'FileSearch':
        """
        Adds one or several filters that match files to be included.

        :param filter_functions:    The filters to be added
        :return:                    The `FileSearch` itself
        """
        self.filters.extend(filter_functions)
        return self

    def filter_by_name(self, *names: str) -> 'FileSearch':
        """
        Adds one or several filters that match files to be included based on their name.

        :param names:   The names of the files that should be included (including their suffix)
        :return:        The `FileSearch` itself
        """

        def filter_file(filtered_names: Set[str], _: str, file_name: str):
            return file_name in filtered_names

        return self.add_filters(*[partial(filter_file, name) for name in names])

    def filter_by_substrings(self,
                             starts_with: Optional[str] = None,
                             not_starts_with: Optional[str] = None,
                             ends_with: Optional[str] = None,
                             not_ends_with: Optional[str] = None,
                             contains: Optional[str] = None,
                             not_contains: Optional[str] = None) -> 'FileSearch':
        """
        Adds a filter that matches files based on whether their name contains specific substrings.

        :param starts_with:     A substring, names must start with or None, if no restrictions should be imposed
        :param not_starts_with: A substring, names must not start with or None, if no restrictions should be imposed
        :param ends_with:       A substring, names must end with or None, if no restrictions should be imposed
        :param not_ends_with:   A substring, names must not end with or None, if no restrictions should be imposed
        :param contains:        A substring, names must contain or None, if no restrictions should be imposed
        :param not_contains:    A substring, names must not contain or None, if no restrictions should be imposed
        :return:                The `FileSearch` itself
        """

        def filter_file(start: Optional[str], not_start: Optional[str], end: Optional[str], not_end: Optional[str],
                        substring: Optional[str], not_substring: Optional[str], _: str, file_name: str):
            return (not start or file_name.startswith(start)) \
                and (not not_start or not file_name.startswith(not_start)) \
                and (not end or file_name.endswith(end)) \
                and (not not_end or file_name.endswith(not_end)) \
                and (not substring or file_name.find(substring) >= 0) \
                and (not not_substring or file_name.find(not_substring) < 0)

        return self.add_filters(
            partial(filter_file, starts_with, not_starts_with, ends_with, not_ends_with, contains, not_contains))

    def filter_by_suffix(self, *suffixes: str) -> 'FileSearch':
        """
        Adds one or several filters that match files to be included based on their suffix.

        :param suffixes:    The suffixes of the files that should be included (without starting dot)
        :return:            The `FileSearch` itself
        """

        def filter_file(filtered_suffixes: List[str], _: str, file_name: str):
            return reduce(lambda aggr, suffix: aggr or file_name.endswith(suffix), filtered_suffixes, False)

        return self.add_filters(partial(filter_file, list(suffixes)))

    def filter_by_language(self, *languages: Language) -> 'FileSearch':
        """
        Adds one or several filters that match files to be included based on the programming language they belong to.

        :param languages:   The languages of the files that should be included
        :return:            The `FileSearch` itself
        """
        return self.filter_by_suffix(*reduce(lambda aggr, language: aggr | language.value, languages, set()))

    def list(self, *directories: str) -> List[str]:
        """
        Lists all files that can be found in given directories.

        :param directories: The directories to search for files
        :return:            A list that contains all files that have been found
        """
        result = []
        subdirectories = self.directory_search.list(*directories) if self.directory_search.recursive else []

        def filter_file(file: str) -> bool:
            return path.isfile(file) and (not self.filters or reduce(
                lambda aggr, file_filter: aggr or file_filter(path.dirname(file), path.basename(file)), self.filters,
                False))

        for directory in list(directories) + subdirectories:
            files = [file for file in glob(path.join(directory, '*')) if filter_file(file)]

            if self.hidden:
                files.extend([file for file in glob(path.join(directory, '.*')) if filter_file(file)])

            result.extend(files)

        return result
