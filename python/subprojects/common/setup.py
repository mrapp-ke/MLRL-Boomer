"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import shutil

from pathlib import Path
from typing import List

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class PrecompiledExtension(Extension):
    """
    Represents a pre-compiled extension module.
    """

    def __init__(self, name: str, path: Path):
        """
        :param name:    The name of the extension module
        :param path:    The path to the extension module
        """
        super().__init__(name, [])
        self.name = name
        self.path = path


class PrecompiledExtensionBuilder(build_ext):
    """
    Copies pre-compiled extension modules into the build directory.
    """

    def build_extension(self, ext):
        """
        See :func:`setuptools.command.build_ext.build_extension`
        """
        if isinstance(ext, PrecompiledExtension):
            build_dir = Path(self.get_ext_fullpath(ext.name)).parent
            target_file = build_dir / ext.path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ext.path, target_file)
        else:
            super().build_extension(ext)


def find_extensions(directory: Path) -> List[PrecompiledExtension]:
    """
    Finds and returns all pre-compiled extension modules.

    :param directory:   The path to a directory, where the search should be started
    :return:            A list that contains all pre-compiled extension modules that have been found
    """
    extensions = []

    for path in directory.rglob('*'):
        if path.is_file() and any(suffix in path.name for suffix in ('.so', '.pyd', '.dylib', '.lib', '.dll')):
            file_name = path.name
            extension_name = file_name[:file_name.find('.')]
            extensions.append(PrecompiledExtension(name=extension_name, path=path))

    return extensions


setup(packages=find_packages(),
      ext_modules=find_extensions(Path('mlrl')),
      cmdclass={'build_ext': PrecompiledExtensionBuilder})
