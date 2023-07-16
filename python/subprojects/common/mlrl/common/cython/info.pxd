from libcpp.string cimport string


cdef extern from "common/info.hpp" nogil:

    cdef cppclass ILibraryInfo:

        string getLibraryVersion() const

    const ILibraryInfo& getCommonLibraryInfo()


cdef class CppLibraryInfo:
    
    # Attributes:

    cdef const ILibraryInfo* library_info_ptr
