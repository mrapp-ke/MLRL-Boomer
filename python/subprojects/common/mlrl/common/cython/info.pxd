from libcpp.memory cimport unique_ptr
from libcpp.string cimport string


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

        void visitBuildOption(BuildOptionVisitor visitor) const

    unique_ptr[ILibraryInfo] getLibraryInfo()


cdef class CppLibraryInfo:
    
    # Attributes:

    cdef unique_ptr[ILibraryInfo] library_info_ptr
