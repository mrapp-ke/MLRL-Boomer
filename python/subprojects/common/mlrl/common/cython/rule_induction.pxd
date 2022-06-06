from mlrl.common.cython._types cimport uint32

from libcpp cimport bool


cdef extern from "common/rule_induction/rule_induction_top_down_greedy.hpp" nogil:

    cdef cppclass IGreedyTopDownRuleInductionConfig:

        # Functions:

        IGreedyTopDownRuleInductionConfig& setMinCoverage(uint32 minCoverage) except +

        uint32 getMinCoverage() const

        IGreedyTopDownRuleInductionConfig& setMaxConditions(uint32 maxConditions) except +

        uint32 getMaxConditions() const;

        IGreedyTopDownRuleInductionConfig& setMaxHeadRefinements(uint32 maxHeadRefinements) except +

        uint32 getMaxHeadRefinements() const

        IGreedyTopDownRuleInductionConfig& setRecalculatePredictions(bool recalculatePredictions) except +

        bool arePredictionsRecalculated() const


cdef class GreedyTopDownRuleInductionConfig:

    # Attributes:

    cdef IGreedyTopDownRuleInductionConfig* config_ptr
