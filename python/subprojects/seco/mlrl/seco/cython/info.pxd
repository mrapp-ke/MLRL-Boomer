from mlrl.common.cython.info cimport ILibraryInfo


cdef extern from "seco/info.hpp" namespace "seco" nogil:

    const ILibraryInfo& getSeCoLibraryInfo()
