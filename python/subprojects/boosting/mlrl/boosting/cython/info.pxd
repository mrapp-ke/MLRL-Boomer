from mlrl.common.cython.info cimport ILibraryInfo


cdef extern from "boosting/info.hpp" namespace "boosting" nogil:

    const ILibraryInfo& getBoostingLibraryInfo()
