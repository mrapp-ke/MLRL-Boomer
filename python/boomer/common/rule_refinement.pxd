"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to find the best refinement of rules.
"""
from boomer.common._arrays cimport uint32, intp, float32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.rules cimport Comparator
from boomer.common.statistics cimport AbstractStatistics
from boomer.common.head_refinement cimport IHeadRefinement

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "cpp/rule_refinement.h" nogil:

    cdef cppclass Refinement:

        # Attributes:

        unique_ptr[PredictionCandidate] headPtr

        uint32 featureIndex

        float32 threshold

        Comparator comparator

        bool covered

        uint32 coveredWeights

        intp start

        intp end

        intp previous


    cdef cppclass AbstractRuleRefinement:

        # Attributes:

        unique_ptr[Refinement] bestRefinementPtr_

        # Functions:

        void findRefinement(IHeadRefinement& headRefinement, PredictionCandidate* currentHead, uint32 numLabelIndices,
                            const uint32* labelIndices)
