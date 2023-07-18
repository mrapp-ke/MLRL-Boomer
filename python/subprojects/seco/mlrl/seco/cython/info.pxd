from libcpp.memory cimport unique_ptr

from mlrl.common.cython.info cimport ILibraryInfo


cdef extern from "seco/info.hpp" namespace "seco" nogil:

    unique_ptr[ILibraryInfo] getLibraryInfo()
