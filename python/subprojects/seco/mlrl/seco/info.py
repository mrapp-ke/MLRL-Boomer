"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python module.
"""
from mlrl.common.cython.info import CppLibraryInfo

from mlrl.seco.cython.info import get_seco_cpp_library_info


def get_cpp_library_info() -> CppLibraryInfo:
    """
    Returns information about the C++ library that is used by this Python package.

    :return: A `CppLibraryInfo` that provides information about the C++ library
    """
    return get_seco_cpp_library_info()
