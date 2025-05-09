"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move

from mlrl.common.cython.package_info cimport CppLibraryInfo


def get_cpp_library_info() -> CppLibraryInfo:
    """
    Returns information about the C++ library that is wrapped by this Cython package.

    :return: A `CppLibraryInfo` that provides information about the C++ library
    """
    cdef CppLibraryInfo library_info = CppLibraryInfo()
    library_info.library_info_ptr = move(getLibraryInfo())
    return library_info
