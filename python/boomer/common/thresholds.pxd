from boomer.common._arrays cimport uint32
from boomer.common.rule_refinement cimport IRuleRefinement
from boomer.common.sub_sampling cimport IWeightVector


cdef extern from "cpp/thresholds.h" nogil:

    cdef cppclass IThresholdsSubset:

        # Functions:

        IRuleRefinement* createRuleRefinement(uint32 featureIndex)


    cdef cppclass AbstractThresholds:

        # Functions:

        uint32 getNumRows()

        uint32 getNumCols()

        IThresholdsSubset* createSubset(IWeightVector* weights)
