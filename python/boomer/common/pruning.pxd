from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport AbstractPrediction
from boomer.common._tuples cimport IndexedFloat32Array
from boomer.common.rules cimport Condition
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.sub_sampling cimport IWeightVector
from boomer.common.thresholds cimport IThresholdsSubset, CoverageMask
from boomer.common.head_refinement cimport IHeadRefinement

from libcpp.list cimport list as double_linked_list
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map


cdef extern from "cpp/pruning.h" nogil:

    cdef cppclass IPruning:

        # Functions:

        const CoverageMask& prune(IThresholdsSubset& thresholdsSubset, double_linked_list[Condition]& conditions,
                                  const AbstractPrediction& head)


    cdef cppclass NoPruningImpl(IPruning):
        pass


    cdef cppclass IREPImpl(IPruning):
        pass


cdef class Pruning:

    # Functions:

    cdef pair[uint32[::1], uint32] prune(self, unordered_map[uint32, IndexedFloat32Array*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, AbstractPrediction* head,
                                         uint32[::1] covered_examples_mask, uint32 covered_examples_target,
                                         IWeightVector* weights, AbstractStatistics* statistics,
                                         IHeadRefinement* head_refinement)


cdef class IREP(Pruning):

    # Functions:

    cdef pair[uint32[::1], uint32] prune(self, unordered_map[uint32, IndexedFloat32Array*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, AbstractPrediction* head,
                                         uint32[::1] covered_examples_mask, uint32 covered_examples_target,
                                         IWeightVector* weights, AbstractStatistics* statistics,
                                         IHeadRefinement* head_refinement)
