"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities for dealing with Meson wrap files.
"""

from util.io import TextFile
from util.version import Version


class WrapFile(TextFile):
    """
    A Meson wrap file.
    """

    KEY_URL = 'url'

    KEY_VERSION = 'revision'

    def __get_value(self, key: str) -> str:
        value = next(
            (line.lstrip(key).lstrip().lstrip('=').lstrip().rstrip() for line in self.lines if line.startswith(key)),
            None,
        )

        if not value:
            raise ValueError(f'No value for key "{key}" found in wrap file {self}')

        return value

    @property
    def dependency_name(self) -> str:
        """
        The name of the dependency declared in the wrap file.
        """
        return self.file.stem

    @property
    def repository_name(self) -> str:
        """
        The name of the repository, the dependency is downloaded from.
        """
        domain = 'https://github.com'
        url = self.__get_value(self.KEY_URL)

        if not url.startswith(domain):
            raise ValueError(f'Expected URL in wrap file {self} to start with "{domain}", but is: {url}')

        return url.lstrip(domain).lstrip('/').rstrip('.git')

    @property
    def version(self) -> Version:
        """
        The version declared in the wrap file.
        """
        return Version.parse(self.__get_value(self.KEY_VERSION))

    def update_version(self, updated_version: Version) -> None:
        self.write_lines(
            *(
                f'{self.KEY_VERSION} = {updated_version}\n' if line.startswith(self.KEY_VERSION) else line
                for line in self.lines
            )
        )
