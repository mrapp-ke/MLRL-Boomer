from libcpp.memory cimport unique_ptr

from mlrl.common.cython.package_info cimport ILibraryInfo


cdef extern from "mlrl/seco/library_info.hpp" namespace "seco" nogil:

    unique_ptr[ILibraryInfo] getLibraryInfo()
