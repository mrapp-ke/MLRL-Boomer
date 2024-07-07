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


def find_dependencies(requirements_file, dependency_names):
    """
    Finds and returns dependencies with given names.

    :param requirements_file:   The path to the requirements.txt file where the dependency versions are specified
    :param dependency_names:    A list that contains the names of the dependencies to be found
    :return:                    A list that contains all dependencies that have been found
    """
    requirements = {line.split(' ')[0]: line for line in requirements_file.read_text().split('\n')}
    dependencies = []

    for dependency_name in dependency_names:
        match = requirements.get(dependency_name)

        if match is None:
            raise RuntimeError('Failed to determine required version of dependency "' + dependency_name + '"')

        dependencies.append(str(match))

    return dependencies


setup(name='mlrl-common',
      version=VERSION,
      description='Provides common modules to be used by different types of multi-label rule learning algorithms',
      long_description=(Path(__file__).resolve().parent / 'README.md').read_text(),
      long_description_content_type='text/markdown',
      author='Michael Rapp',
      author_email='michael.rapp.ml@gmail.com',
      url='https://github.com/mrapp-ke/MLRL-Boomer',
      download_url='https://github.com/mrapp-ke/MLRL-Boomer/releases',
      project_urls={
          'Documentation': 'https://mlrl-boomer.readthedocs.io/en/latest',
          'Issue Tracker': 'https://github.com/mrapp-ke/MLRL-Boomer/issues',
      },
      license='MIT',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: C++',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 3',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords=[
          'machine learning',
          'scikit-learn',
          'multi-label classification',
          'rule learning',
      ],
      platforms=[
          'Linux',
          'MacOS',
          'Windows',
      ],
      python_requires=PYTHON_VERSION,
      install_requires=[
          find_dependencies(requirements_file=Path(__file__).resolve().parent.parent.parent / 'requirements.txt',
                            dependency_names=['numpy', 'scipy', 'scikit-learn']),
      ],
      packages=find_packages(),
      ext_modules=find_extensions('mlrl'),
      cmdclass={'build_ext': PrecompiledExtensionBuilder},
      zip_safe=True)
