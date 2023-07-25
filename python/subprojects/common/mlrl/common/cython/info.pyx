"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move

from dataclasses import dataclass
from typing import List


@dataclass
class CppBuildOption:
    """
    Represents a build option for configuring a C++ library at compile-time.

    Attributes:
        option:         The name of the build option
        description:    A human-legible description of the build option
        value:          The value that has been set for the build option at compile-time
    """
    option: str
    description: str
    value: str

    def __str__(self) -> str:
        return self.description + ' (' + self.value + ')'


cdef class CppLibraryInfo:
    """
    Provides information about a C++ library.
    """

    cdef __visit_build_option(self, const BuildOption& build_option):
        cdef string option = build_option.option
        cdef string description = build_option.description
        cdef string value = build_option.value
        self.__build_options.append(CppBuildOption(option=option.decode('UTF-8'),
                                                   description=description.decode('UTF-8'),
                                                   value=value.decode('UTF-8')))

    @property
    def library_name(self) -> str:
        """
        The name of the C++ library.
        """
        cdef string library_name = self.library_info_ptr.get().getLibraryName()
        return library_name.decode('UTF-8')

    @property
    def library_version(self) -> str:
        """
        The version of the C++ library.
        """
        cdef string library_version = self.library_info_ptr.get().getLibraryVersion()
        return library_version.decode('UTF-8')

    @property
    def target_architecture(self) -> str:
        """
        The architecture that is targeted by the C++ library.
        """
        cdef string target_architecture = self.library_info_ptr.get().getTargetArchitecture()
        return target_architecture.decode('UTF-8')

    @property
    def build_options(self) -> List[CppBuildOption]:
        """
        The build options for configuring the C++ library at compile-time.
        """
        self.__build_options = []
        self.library_info_ptr.get().visitBuildOptions(
            wrapBuildOptionVisitor(<void*>self, <BuildOptionCythonVisitor>self.__visit_build_option))
        return self.__build_options

    def __str__(self) -> str:
        return self.library_name + ' ' + self.library_version + ' (' + self.target_architecture + ')'

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


def get_cpp_library_info() -> CppLibraryInfo:
    """
    Returns information about the C++ library that is wrapped by this Cython package.

    :return: A `CppLibraryInfo` that provides information about the C++ library
    """
    cdef CppLibraryInfo library_info = CppLibraryInfo()
    library_info.library_info_ptr = move(getLibraryInfo())
    return library_info
