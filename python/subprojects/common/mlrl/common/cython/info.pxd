from libcpp.string cimport string


cdef extern from "common/info.hpp" nogil:

    cdef cppclass ILibraryInfo:

        string getLibraryName() const

        string getLibraryVersion() const

        string getTargetArchitecture() const

    const ILibraryInfo& getCommonLibraryInfo()


cdef class CppLibraryInfo:
    
    # Attributes:

    cdef const ILibraryInfo* library_info_ptr
