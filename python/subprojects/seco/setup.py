"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
import os
import shutil

from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

VERSION = (Path(__file__).resolve().parent.parent.parent.parent / '.version').read_text()

PYTHON_VERSION = (Path(__file__).resolve().parent.parent.parent.parent / '.version-python').read_text()


class PrecompiledExtension(Extension):
    """
    Represents a pre-compiled extension module.
    """

    def __init__(self, name: str, path: Path):
        """
        :param name:    The name of the extension module
        :param path:    The path of the extension module
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
            target_file = Path(os.path.join(build_dir, ext.path))
            os.makedirs(target_file.parent, exist_ok=True)
            shutil.copy(ext.path, target_file)
        else:
            super().build_extension(ext)


def find_extensions(directory):
    """
    Finds and returns all pre-compiled extension modules.

    :return: A list that contains all pre-comiled extension modules that have been found
    """
    extensions = []

    for path, _, file_names in os.walk(directory):
        for file_name in file_names:
            if '.so' in file_name or '.pyd' in file_name or '.dylib' in file_name or '.lib' in file_name \
                    or '.dll' in file_name:
                extension_path = Path(os.path.join(path, file_name))
                extension_name = file_name[:file_name.find('.')]
                extensions.append(PrecompiledExtension(extension_name, extension_path))

    return extensions


setup(version=VERSION,
      python_requires=PYTHON_VERSION,
      install_requires=[
          'mlrl-common==' + VERSION,
      ],
      extras_require={
          'MLRL_TESTBED': ['mlrl-testbed==' + VERSION],
      },
      packages=find_packages(),
      ext_modules=find_extensions('mlrl'),
      cmdclass={'build_ext': PrecompiledExtensionBuilder})
