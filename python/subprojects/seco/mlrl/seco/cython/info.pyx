"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.info cimport CppLibraryInfo


def get_seco_cpp_library_info() -> CppLibraryInfo:
    """
    Returns information about the C++ library "libmlrlseco".

    :return: A `CppLibraryInfo` that provides information about the C++ library
    """
    cdef CppLibraryInfo library_info = CppLibraryInfo()
    library_info.library_info_ptr = &getSeCoLibraryInfo()
    return library_info
