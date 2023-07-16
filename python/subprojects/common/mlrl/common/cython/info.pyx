"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from cython.operator cimport dereference


cdef class LibraryInfo:
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


def get_common_library_info() -> LibraryInfo:
    """
    Returns information about the C++ library "libmlrlcommon".

    :return: A `LibraryInfo` that provides information about the C++ library
    """
    cdef LibraryInfo library_info = LibraryInfo()
    library_info.library_info_ptr = &getCommonLibraryInfo()
    return library_info
