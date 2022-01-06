from mlrl.common.cython._types cimport uint32

from libcpp cimport bool


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass TopDownRuleInductionConfigImpl"TopDownRuleInductionConfig":

        # Functions:

        TopDownRuleInductionConfigImpl& setMinCoverage(uint32 minCoverage)

        TopDownRuleInductionConfigImpl& setMaxConditions(uint32 maxConditions)

        TopDownRuleInductionConfigImpl& setMaxHeadRefinements(uint32 maxHeadRefinements)

        TopDownRuleInductionConfigImpl& setRecalculatePredictions(bool recalculatePredictions)

        TopDownRuleInductionConfigImpl& setNumThreads(uint32 numThreads)


cdef class TopDownRuleInductionConfig:

    # Attributes:

    cdef TopDownRuleInductionConfigImpl* config_ptr
