from libcpp.memory cimport unique_ptr
from libcpp.string cimport string


cdef extern from "common/info.hpp" nogil:

    cdef cppclass ILibraryInfo:

        string getLibraryName() const

        string getLibraryVersion() const

        string getTargetArchitecture() const

    unique_ptr[ILibraryInfo] getCommonLibraryInfo()


cdef class CppLibraryInfo:
    
    # Attributes:

    cdef unique_ptr[ILibraryInfo] library_info_ptr
