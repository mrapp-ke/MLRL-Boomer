"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class CppLibraryInfo:
    """
    Provides information about a C++ library.
    """

    def get_library_name(self) -> str:
        """
        Returns the name of the C++ library.

        :return: A string that specifies the library name
        """
        cdef string library_name = self.library_info_ptr.getLibraryName()
        return library_name.decode('UTF-8')

    def get_library_version(self) -> str:
        """
        Returns the version of the C++ library.

        :return: A string that specifies the library version
        """
        cdef string library_version = self.library_info_ptr.getLibraryVersion()
        return library_version.decode('UTF-8')

    def get_target_architecture(self) -> str:
        """
        Returns the architecture that is targeted by the C++ library.

        :return: A string that specifies the target architecture
        """
        cdef string target_architecture = self.library_info_ptr.getTargetArchitecture()
        return target_architecture.decode('UTF-8')

    def __str__(self) -> str:
        return self.get_library_name() + ' ' + self.get_library_version() + ' (' + self.get_target_architecture() + ')'


def get_common_cpp_library_info() -> CppLibraryInfo:
    """
    Returns information about the C++ library "libmlrlcommon".

    :return: A `CppLibraryInfo` that provides information about the C++ library
    """
    cdef CppLibraryInfo library_info = CppLibraryInfo()
    library_info.library_info_ptr = &getCommonLibraryInfo()
    return library_info
