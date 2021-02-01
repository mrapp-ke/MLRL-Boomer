from boomer.common._types cimport uint32, intp
from boomer.common._indices cimport IIndexVector
from boomer.common.model cimport IModelBuilder
from boomer.common.statistics cimport IStatisticsProvider
from boomer.common.thresholds cimport IThresholds
from boomer.common.sampling cimport IWeightVector, IFeatureSubSampling, RNG
from boomer.common.pruning cimport IPruning
from boomer.common.post_processing cimport IPostProcessor
from boomer.common.head_refinement cimport IHeadRefinementFactory

from libcpp cimport bool
from libcpp.memory cimport shared_ptr


cdef extern from "cpp/rule_induction/rule_induction.h" nogil:

    cdef cppclass IRuleInduction:

        # Functions:

        void induceDefaultRule(IStatisticsProvider& statisticsProvider,
                               const IHeadRefinementFactory* headRefinementFactory, IModelBuilder& modelBuilder)

        bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                        const IFeatureSubSampling& featureSubSampling, const IPruning& pruning,
                        const IPostProcessor& postProcessor, uint32 minCoverage, intp maxConditions,
                        intp maxHeadRefinements, int numThreads, RNG& rng, IModelBuilder& modelBuilder)


cdef extern from "cpp/rule_induction/rule_induction_top_down.h" nogil:

    cdef cppclass TopDownRuleInductionImpl"TopDownRuleInduction"(IRuleInduction):
        pass


cdef class RuleInduction:

    # Attributes:

    cdef shared_ptr[IRuleInduction] rule_induction_ptr


cdef class TopDownRuleInduction(RuleInduction):
    pass
