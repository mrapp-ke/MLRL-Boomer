from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from mlrl.common.cython._types cimport uint32


cdef extern from "common/info.hpp" nogil:

    cdef cppclass BuildOption"ILibraryInfo::BuildOption":

        string option

        string description

        string value


ctypedef void (*BuildOptionVisitor)(const BuildOption&)


cdef extern from "common/info.hpp" nogil:

    cdef cppclass ILibraryInfo:

        string getLibraryName() const

        string getLibraryVersion() const

        string getTargetArchitecture() const

        void visitBuildOptions(BuildOptionVisitor visitor) const

    unique_ptr[ILibraryInfo] getLibraryInfo()

    bool isMultiThreadingSupportEnabled()

    uint32 getNumCpuCores()


cdef extern from *:
    """
    #include "common/info.hpp"


    typedef void (*BuildOptionCythonVisitor)(void*, const ILibraryInfo::BuildOption&);

    static inline ILibraryInfo::BuildOptionVisitor wrapBuildOptionVisitor(void* self,
                                                                          BuildOptionCythonVisitor visitor) {
        return [=](const ILibraryInfo::BuildOption& buildOption) {
            visitor(self, buildOption);
        };
    }
    """

    ctypedef void (*BuildOptionCythonVisitor)(void*, const BuildOption&)

    BuildOptionVisitor wrapBuildOptionVisitor(void* self, BuildOptionCythonVisitor visitor)


cdef class CppLibraryInfo:
    
    # Attributes:

    cdef unique_ptr[ILibraryInfo] library_info_ptr

    cdef list __build_options

    # Functions:

    cdef __visit_build_option(self, const BuildOption& build_option)
