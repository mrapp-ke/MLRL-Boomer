"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path

from setuptools import find_packages, setup

VERSION = (Path(__file__).resolve().parent.parent.parent.parent / '.version').read_text()

PYTHON_VERSION = (Path(__file__).resolve().parent.parent.parent.parent / '.version-python').read_text()


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

        dependencies.append(match)

    return dependencies


setup(version=VERSION,
      python_requires=PYTHON_VERSION,
      install_requires=[
          'mlrl-common==' + VERSION,
          *find_dependencies(requirements_file=Path(__file__).resolve().parent / 'requirements.txt',
                             dependency_names=['liac-arff', 'tabulate']),
      ],
      extras_require={
          'BOOMER': ['mlrl-boomer==' + VERSION],
          'SECO': ['mlrl-seco==' + VERSION],
      },
      packages=find_packages())
