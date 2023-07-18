from libcpp.memory cimport unique_ptr

from mlrl.common.cython.info cimport ILibraryInfo


cdef extern from "boosting/info.hpp" namespace "boosting" nogil:

    unique_ptr[ILibraryInfo]  getLibraryInfo()
