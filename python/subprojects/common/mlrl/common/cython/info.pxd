from libcpp.string cimport string


cdef extern from "common/info.hpp" nogil:

    string getLibraryVersion()
