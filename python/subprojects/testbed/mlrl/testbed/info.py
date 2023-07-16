"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python package.
"""
from mlrl.common.cython.info import get_common_cpp_library_info
from mlrl.common.info import PythonPackageInfo


def get_package_info() -> PythonPackageInfo:
    """
    Returns information about this Python package.

    :return: A `PythonPackageInfo` that provides information about the Python package
    """
    return PythonPackageInfo('mlrl-testbed', cpp_libraries=[get_common_cpp_library_info()])
