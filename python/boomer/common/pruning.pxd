from boomer.common._predictions cimport AbstractPrediction
from boomer.common.rules cimport Condition, ConditionList
from boomer.common.thresholds cimport IThresholdsSubset, CoverageMask

from libcpp.memory cimport unique_ptr, shared_ptr


cdef extern from "cpp/pruning.h" nogil:

    cdef cppclass IPruning:

        # Functions:

        unique_ptr[CoverageMask] prune(IThresholdsSubset& thresholdsSubset, ConditionList& conditions,
                                       const AbstractPrediction& head)


    cdef cppclass NoPruningImpl(IPruning):
        pass


    cdef cppclass IREPImpl(IPruning):
        pass


cdef class Pruning:

    # Attributes:

    cdef shared_ptr[IPruning] pruning_ptr


cdef class NoPruning(Pruning):
    pass


cdef class IREP(Pruning):
    pass
