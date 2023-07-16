"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from cython.operator cimport dereference


cdef class CppLibraryInfo:
    """
    Provides information about a C++ library.
    """

    def get_library_version(self) -> str:
        """
        Returns the version of the C++ library.

        :return: A string that specifies the version
        """
        cdef string library_version = self.library_info_ptr.getLibraryVersion()
        return library_version.decode('UTF-8')


def get_common_cpp_library_info() -> CppLibraryInfo:
    """
    Returns information about the C++ library "libmlrlcommon".

    :return: A `LibraryInfo` that provides information about the C++ library
    """
    cdef CppLibraryInfo library_info = CppLibraryInfo()
    library_info.library_info_ptr = &getCommonLibraryInfo()
    return library_info
