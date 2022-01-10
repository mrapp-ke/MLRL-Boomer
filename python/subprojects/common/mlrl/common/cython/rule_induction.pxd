from mlrl.common.cython._types cimport uint32

from libcpp cimport bool


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass TopDownRuleInductionConfigImpl"TopDownRuleInductionConfig":

        # Functions:

        TopDownRuleInductionConfigImpl& setMinCoverage(uint32 minCoverage) except +

        TopDownRuleInductionConfigImpl& setMaxConditions(uint32 maxConditions) except +

        TopDownRuleInductionConfigImpl& setMaxHeadRefinements(uint32 maxHeadRefinements) except +

        TopDownRuleInductionConfigImpl& setRecalculatePredictions(bool recalculatePredictions) except +

        TopDownRuleInductionConfigImpl& setNumThreads(uint32 numThreads) except +


cdef class TopDownRuleInductionConfig:

    # Attributes:

    cdef TopDownRuleInductionConfigImpl* config_ptr
