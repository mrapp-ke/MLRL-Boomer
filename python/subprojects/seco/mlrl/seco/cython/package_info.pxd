from libcpp.memory cimport unique_ptr

from mlrl.common.cython.package_info cimport ILibraryInfo


cdef extern from "mlrl/seco/info.hpp" namespace "seco" nogil:

    unique_ptr[ILibraryInfo] getLibraryInfo()
