from mlrl.common.cython._types cimport uint32

from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "common/rule_induction/rule_induction.hpp" nogil:

    cdef cppclass IRuleInductionFactory:
        pass


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass TopDownRuleInductionFactoryImpl"TopDownRuleInductionFactory"(IRuleInductionFactory):

        # Constructors:

        TopDownRuleInductionFactoryImpl(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                        bool recalculatePredictions, uint32 numThreads) except +


cdef class RuleInductionFactory:

    # Attributes:

    cdef unique_ptr[IRuleInductionFactory] rule_induction_factory_ptr


cdef class TopDownRuleInductionFactory(RuleInductionFactory):
    pass
