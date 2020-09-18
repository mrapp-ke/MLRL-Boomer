"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to find the best refinement of rules.
"""
from boomer.common._arrays cimport uint32, intp, float32
from boomer.common._tuples cimport IndexedFloat32Array, IndexedFloat32ArrayWrapper
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.rules cimport Comparator
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.head_refinement cimport AbstractHeadRefinement

from libcpp cimport bool


cdef extern from "cpp/rule_refinement.h" nogil:

    cdef struct Refinement:
        PredictionCandidate* head
        uint32 featureIndex
        float32 threshold
        Comparator comparator
        bool covered
        uint32 coveredWeights
        intp start
        intp end
        intp previous
        IndexedFloat32Array* indexedArray
        IndexedFloat32ArrayWrapper* indexedArrayWrapper


    cdef cppclass AbstractRuleRefinement:

        # Functions:

        Refinement findRefinement(AbstractHeadRefinement* headRefinement, PredictionCandidate* currentHead,
                                  uint32 numLabelIndices, const uint32* labelIndices)


    cdef cppclass RuleRefinementImpl(AbstractRuleRefinement):

        # Constructors:

        RuleRefinementImpl(AbstractStatistics* statistics, IndexedFloat32Array* indexedArray, const uint32* weights,
                           uint32 totalSumOfWeights, uint32 featureIndex, bool nominal) except +
