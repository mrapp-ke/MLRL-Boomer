"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python module.
"""
from mlrl.common.cython.info import CppLibraryInfo, get_common_cpp_library_info


def get_cpp_library_info() -> CppLibraryInfo:
    """
    Returns information about the C++ library "libmlrlcommon".

    :return: A `CppLibraryInfo` that provides information about the C++ library
    """
    return get_common_cpp_library_info()
