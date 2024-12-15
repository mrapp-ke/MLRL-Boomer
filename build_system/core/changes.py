"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for detecting changes in files.
"""
import json

from functools import cached_property
from os import path
from typing import Dict, List, Set

from core.modules import Module
from util.io import TextFile, create_directories


class JsonFile(TextFile):
    """
    Allows to read and write the content of a JSON file.
    """

    @cached_property
    def json(self) -> Dict:
        """
        The content of the JSON file as a dictionary.
        """
        lines = self.lines

        if lines:
            return json.loads('\n'.join(lines))

        return {}

    def write_json(self, dictionary: Dict):
        """
        Writes a given dictionary to the JSON file.

        :param dictionary: The dictionary to be written
        """
        self.write_lines(json.dumps(dictionary, indent=4))

    def write_lines(self, *lines: str):
        super().write_lines(*lines)

        try:
            del self.json
        except AttributeError:
            pass


class ChangeDetection:
    """
    Allows to detect changes in tracked files.
    """

    class CacheFile(JsonFile):
        """
        A JSON file that stores checksums for tracked files.
        """

        @staticmethod
        def __checksum(file: str) -> str:
            return str(path.getmtime(file))

        def __init__(self, file: str):
            """
            :param file: The path to the JSON file
            """
            super().__init__(file, accept_missing=True)
            create_directories(path.dirname(file))

        def update(self, module_name: str, files: Set[str]):
            """
            Updates the checksums of given files.

            :param module_name: The name of the module, the files belong to
            :param files:       A set that contains the paths of the files to be updated
            """
            cache = self.json
            module_cache = cache.setdefault(module_name, {})

            for invalid_key in [file for file in module_cache.keys() if file not in files]:
                del module_cache[invalid_key]

            for file in files:
                module_cache[file] = self.__checksum(file)

            if module_cache:
                cache[module_name] = module_cache
            else:
                del cache[module_name]

            if cache:
                self.write_json(cache)
            else:
                self.delete()

        def has_changed(self, module_name: str, file: str) -> bool:
            """
            Returns whether a file has changed according to the cache or not.

            :param module_name: The name of the module, the file belongs to
            :param file:        The file to be checked
            :return:            True, if the file has changed, False otherwise
            """
            module_cache = self.json.get(module_name, {})
            return file not in module_cache or module_cache[file] != self.__checksum(file)

    def __init__(self, cache_file: str):
        """
        :param cache_file: The path to the file that should be used for tracking files
        """
        self.cache_file = ChangeDetection.CacheFile(cache_file)

    def track_files(self, module: Module, *files: str):
        """
        Updates the cache to keep track of given files.

        :param module:  The module, the files belong to
        :param files:   The files to be tracked
        """
        self.cache_file.update(str(module), set(files))

    def get_changed_files(self, module: Module, *files: str) -> List[str]:
        """
        Filters given files and returns only those that have changed.

        :param module:  The module, the files belong to
        :param files:   The files to be filtered
        :return:        A list that contains the files that have changed
        """
        module_name = str(module)
        return [file for file in files if self.cache_file.has_changed(module_name, file)]
