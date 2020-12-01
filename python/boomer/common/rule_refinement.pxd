"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to find the best refinement of rules.
"""
from boomer.common._types cimport uint32, intp, float32
from boomer.common.head_refinement cimport AbstractEvaluatedPrediction
from boomer.common.rules cimport Comparator

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "cpp/rule_refinement/rule_refinement.h" nogil:

    cdef cppclass Refinement:

        # Attributes:

        unique_ptr[AbstractEvaluatedPrediction] headPtr

        uint32 featureIndex

        float32 threshold

        Comparator comparator

        bool covered

        uint32 coveredWeights

        intp start

        intp end

        intp previous

        # Functions:

        bool isBetterThan(Refinement& another)


    cdef cppclass IRuleRefinement:

        # Functions:

        void findRefinement(AbstractEvaluatedPrediction* currentHead)

        unique_ptr[Refinement] pollRefinement()
