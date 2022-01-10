from mlrl.common.cython._types cimport uint32

from libcpp cimport bool


cdef extern from "common/rule_induction/rule_induction_top_down.hpp" nogil:

    cdef cppclass TopDownRuleInductionConfigImpl"TopDownRuleInductionConfig":

        # Functions:

        TopDownRuleInductionConfigImpl& setMinCoverage(uint32 minCoverage) except +

        uint32 getMinCoverage() const

        TopDownRuleInductionConfigImpl& setMaxConditions(uint32 maxConditions) except +

        uint32 getMaxConditions() const;

        TopDownRuleInductionConfigImpl& setMaxHeadRefinements(uint32 maxHeadRefinements) except +

        uint32 getMaxHeadRefinements() const

        TopDownRuleInductionConfigImpl& setRecalculatePredictions(bool recalculatePredictions) except +

        bool getRecalculatePredictions() const

        TopDownRuleInductionConfigImpl& setNumThreads(uint32 numThreads) except +

        uint32 getNumThreads() const


cdef class TopDownRuleInductionConfig:

    # Attributes:

    cdef TopDownRuleInductionConfigImpl* config_ptr
