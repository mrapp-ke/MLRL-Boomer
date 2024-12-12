"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for listing files and directories.
"""
from abc import abstractmethod
from functools import partial, reduce
from glob import glob
from os import path
from typing import Callable, List, Optional, Set


class DirectorySearch:
    """
    Allows to search for subdirectories.
    """

    Filter = Callable[[str, str], bool]

    def __init__(self):
        self.recursive = False
        self.excludes = []
        self.filters = []

    def set_recursive(self, recursive: bool) -> 'DirectorySearch':
        """
        Sets whether the search should be recursive or not.

        :param recursive:   True, if the search should be recursive, False otherwise
        :return:            The `DirectorySearch` itself
        """
        self.recursive = recursive
        return self

    def add_filters(self, *filter_functions: Filter) -> 'DirectorySearch':
        """
        Adds one or several filters that match subdirectories to be included.

        :param filter_functions:    The filters to be added
        :return:                    The `DirectorySearch` itself
        """
        self.filters.extend(filter_functions)
        return self

    def filter_by_name(self, *names: str) -> 'DirectorySearch':
        """
        Adds one or several filters that match subdirectories to be included based on their name.

        :param names:   The names of the subdirectories that should be included
        :return:        The `DirectorySearch` itself
        """

        def filter_directory(filtered_names: Set[str], _: str, directory_name: str):
            return directory_name in filtered_names

        return self.add_filters(*[partial(filter_directory, name) for name in names])

    def exclude(self, *excludes: Filter) -> 'DirectorySearch':
        """
        Adds one or several filters that should be used for excluding subdirectories.

        :param excludes:    The filters to be set
        :return:            The `DirectorySearch` itself
        """
        self.excludes.extend(excludes)
        return self

    def exclude_by_name(self, *names: str) -> 'DirectorySearch':
        """
        Adds one or several filters that should be used for excluding subdirectories by their names.

        :param names:   The names of the subdirectories to be excluded
        :return:        The `DirectorySearch` itself
        """

        def filter_directory(excluded_name: str, _: str, directory_name: str):
            return directory_name == excluded_name

        return self.exclude(*[partial(filter_directory, name) for name in names])

    def exclude_by_substrings(self,
                              starts_with: Optional[str] = None,
                              not_starts_with: Optional[str] = None,
                              ends_with: Optional[str] = None,
                              not_ends_with: Optional[str] = None,
                              contains: Optional[str] = None,
                              not_contains: Optional[str] = None) -> 'DirectorySearch':
        """
        Adds a filter that that should be used for excluding subdirectories based on whether their name contains
        specific substrings.

        :param starts_with:     A substring, names must start with or None, if no restrictions should be imposed
        :param not_starts_with: A substring, names must not start with or None, if no restrictions should be imposed
        :param ends_with:       A substring, names must end with or None, if no restrictions should be imposed
        :param not_ends_with:   A substring, names must not end with or None, if no restrictions should be imposed
        :param contains:        A substring, names must contain or None, if no restrictions should be imposed
        :param not_contains:    A substring, names must not contain or None, if no restrictions should be imposed
        :return:                The `DirectorySearch` itself
        """

        def filter_directory(start: Optional[str], not_start: Optional[str], end: Optional[str], not_end: Optional[str],
                             substring: Optional[str], not_substring: Optional[str], _: str, file_name: str):
            return (not start or file_name.startswith(start)) \
                and (not not_start or not file_name.startswith(not_start)) \
                and (not end or file_name.endswith(end)) \
                and (not not_end or file_name.endswith(not_end)) \
                and (not substring or file_name.find(substring) >= 0) \
                and (not not_substring or file_name.find(not_substring) < 0)

        return self.exclude(
            partial(filter_directory, starts_with, not_starts_with, ends_with, not_ends_with, contains, not_contains))

    def list(self, *directories: str) -> List[str]:
        """
        Lists all subdirectories that can be found in given directories.

        :param directories: The directories to search for subdirectories
        :return:            A list that contains all subdirectories that have been found
        """
        result = []

        def filter_file(file: str) -> bool:
            if path.isdir(file):
                parent = path.dirname(file)
                file_name = path.basename(file)

                if not reduce(lambda aggr, exclude: aggr or exclude(parent, file_name), self.excludes, False):
                    return True

            return False

        def filter_subdirectory(subdirectory: str, filters: List[DirectorySearch.Filter]) -> bool:
            parent = path.dirname(subdirectory)
            directory_name = path.basename(subdirectory)

            if reduce(lambda aggr, dir_filter: aggr or dir_filter(parent, directory_name), filters, False):
                return True

            return False

        for directory in directories:
            subdirectories = [file for file in glob(path.join(directory, '*')) if filter_file(file)]

            if self.recursive:
                result.extend(self.list(*subdirectories))

            if self.filters:
                subdirectories = [
                    subdirectory for subdirectory in subdirectories if filter_subdirectory(subdirectory, self.filters)
                ]

            result.extend(subdirectories)

        return result


class FileSearch:
    """
    Allows to search for files.
    """

    Filter = Callable[[str, str], bool]

    def __init__(self):
        self.hidden = False
        self.symlinks = True
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

    def add_subdirectory_filters(self, *filter_functions: DirectorySearch.Filter) -> 'FileSearch':
        """
        Adds one or several filters that match subdirectories to be included.

        :param filter_functions:    The filters to be added
        :return:                    The `FileSearch` itself
        """
        self.directory_search.add_filters(*filter_functions)
        return self

    def filter_subdirectories_by_name(self, *names: str) -> 'FileSearch':
        """
        Adds one or several filters that match subdirectories to be included based on their name.

        :param names:   The names of the subdirectories that should be included
        :return:        The `FileSearch` itself
        """
        self.directory_search.filter_by_name(*names)
        return self

    def exclude_subdirectories(self, *excludes: DirectorySearch.Filter) -> 'FileSearch':
        """
        Adds one or several filters that should be used for excluding subdirectories. Does only have an effect if the
        search is recursive.

        :param excludes:    The filters to be set
        :return:            The `FileSearch` itself
        """
        self.directory_search.exclude(*excludes)
        return self

    def exclude_subdirectories_by_name(self, *names: str) -> 'FileSearch':
        """
        Adds one or several filters that should be used for excluding subdirectories by their names. Does only have an
        effect if the search is recursive.

        :param names:   The names of the subdirectories to be excluded
        :return:        The `FileSearch` itself
        """
        self.directory_search.exclude_by_name(*names)
        return self

    def exclude_subdirectories_by_substrings(self,
                                             starts_with: Optional[str] = None,
                                             not_starts_with: Optional[str] = None,
                                             ends_with: Optional[str] = None,
                                             not_ends_with: Optional[str] = None,
                                             contains: Optional[str] = None,
                                             not_contains: Optional[str] = None) -> 'FileSearch':
        """
        Adds a filter that should be used for excluding subdirectories based on whether their name contains specific
        substrings.

        :param starts_with:     A substring, names must start with or None, if no restrictions should be imposed
        :param not_starts_with: A substring, names must not start with or None, if no restrictions should be imposed
        :param ends_with:       A substring, names must end with or None, if no restrictions should be imposed
        :param not_ends_with:   A substring, names must not end with or None, if no restrictions should be imposed
        :param contains:        A substring, names must contain or None, if no restrictions should be imposed
        :param not_contains:    A substring, names must not contain or None, if no restrictions should be imposed
        :return:                The `FileSearch` itself
        """
        self.directory_search.exclude_by_substrings(starts_with=starts_with,
                                                    not_starts_with=not_starts_with,
                                                    ends_with=ends_with,
                                                    not_ends_with=not_ends_with,
                                                    contains=contains,
                                                    not_contains=not_contains)
        return self

    def set_hidden(self, hidden: bool) -> 'FileSearch':
        """
        Sets whether hidden files should be included or not.

        :param hidden:  True, if hidden files should be included, False otherwise
        :return:        The `FileSearch` itself
        """
        self.hidden = hidden
        return self

    def set_symlinks(self, symlinks: bool) -> 'FileSearch':
        """
        Sets whether symbolic links should be included or not.

        :param symlinks:    True, if symbolic links should be included, False otherwise
        :return:            The `FileSearch` itself
        """
        self.symlinks = symlinks
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

    def filter_by_file_type(self, *file_types: 'FileType') -> 'FileSearch':
        """
        Adds one or several filters that match files to be included based on a `FileType`.

        :param file_types:  The `FileType` of the files that should be included
        :return:            The `FileSearch` itself
        """
        for file_type in file_types:
            file_type.file_search_decorator(self)

        return self

    def list(self, *directories: str) -> List[str]:
        """
        Lists all files that can be found in given directories.

        :param directories: The directories to search for files
        :return:            A list that contains all files that have been found
        """
        result = []
        subdirectories = self.directory_search.list(*directories) if self.directory_search.recursive else []

        def filter_file(file: str) -> bool:
            if path.isfile(file) and (self.symlinks or not path.islink(file)):
                if not self.filters:
                    return True

                parent = path.dirname(file)
                file_name = path.basename(file)

                if reduce(lambda aggr, file_filter: aggr or file_filter(parent, file_name), self.filters, False):
                    return True

            return False

        for directory in list(directories) + subdirectories:
            files = [file for file in glob(path.join(directory, '*')) if filter_file(file)]

            if self.hidden:
                files.extend([file for file in glob(path.join(directory, '.*')) if filter_file(file)])

            result.extend(files)

        return result


class FileType:
    """
    Represents different types of files.
    """

    def __init__(self,
                 name: str,
                 suffixes: Set[str],
                 file_search_decorator: Optional[Callable[[FileSearch], None]] = None):
        """
        :param name:                    The name of the file type
        :param suffixes:                The suffixes that correspond to this file type (without leading dot)
        :param file_search_decorator:   A function that adds a filter for this file type to a `FileSearch` or None, if a
                                        filter should automatically be created
        """
        self.name = name
        self.suffixes = suffixes
        self.file_search_decorator = file_search_decorator if file_search_decorator else lambda file_search: file_search.filter_by_suffix(
            *suffixes)

    @staticmethod
    def python() -> 'FileType':
        """
        Creates and returns a `FileType` that corresponds to Python source files.

        :return: The `FileType` that has been created
        """
        return FileType(name='Python', suffixes={'py'})

    @staticmethod
    def cpp() -> 'FileType':
        """
        Creates and returns a `FileType` that corresponds to C++ source files.

        :return: The `FileType` that has been created
        """
        return FileType(name='C++', suffixes={'cpp', 'hpp'})

    @staticmethod
    def cython() -> 'FileType':
        """
        Creates and returns a `FileType` that corresponds to Cython source files.

        :return: The `FileType` that has been created
        """
        return FileType(name='Cython', suffixes={'pyx', 'pxd'})

    @staticmethod
    def markdown() -> 'FileType':
        """
        Creates and returns a `FileType` that corresponds to Markdown files.

        :return: The `FileType` that has been created
        """
        return FileType(name='Markdown', suffixes={'md'})

    @staticmethod
    def yaml() -> 'FileType':
        """
        Creates and returns a `FileType` that corresponds to YAML files.

        :return: The `FileType` that has been created
        """
        return FileType(name='YAML', suffixes={'yaml', 'yml'})

    @staticmethod
    def extension_module() -> 'FileType':
        """
        Creates and returns a `FileType` that corresponds to shared libraries.

        :return: The `FileType` that has been created
        """
        return FileType(
            name='Extension module',
            suffixes={'so', 'pyd', 'lib'},
            file_search_decorator=lambda file_search: file_search \
                .filter_by_substrings(not_starts_with='lib', ends_with='.so') \
                .filter_by_substrings(ends_with='.pyd') \
                .filter_by_substrings(not_starts_with='mlrl', ends_with='.lib'),
        )

    @staticmethod
    def shared_library() -> 'FileType':
        """
        Creates and returns a `FileType` that corresponds to shared libraries.

        :return: The `FileType` that has been created
        """
        return FileType(
            name='Shared library',
            suffixes={'so', 'dylib', 'lib', 'dll'},
            file_search_decorator=lambda file_search: file_search \
                .filter_by_substrings(starts_with='lib', contains='.so') \
                .filter_by_substrings(ends_with='.dylib') \
                .filter_by_substrings(starts_with='mlrl', ends_with='.lib') \
                .filter_by_substrings(ends_with='.dll'),
        )

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: 'FileType') -> bool:
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
