from libcpp.memory cimport unique_ptr

from mlrl.common.cython.package_info cimport ILibraryInfo


cdef extern from "mlrl/boosting/library_info.hpp" namespace "boosting" nogil:

    unique_ptr[ILibraryInfo]  getLibraryInfo()
