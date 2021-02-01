/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_induction.h"


/**
 * Allows to induce classification rules using a top-down greedy search, where new conditions are added iteratively to
 * the (initially empty) body of a rule. At each iteration, the refinement that improves the rule the most is chosen.
 * The search stops if no refinement results in an improvement.
 */
class TopDownRuleInduction : public IRuleInduction {

    public:

        void induceDefaultRule(IStatisticsProvider& statisticsProvider,
                               const IHeadRefinementFactory* headRefinementFactory,
                               IModelBuilder& modelBuilder) const override;

        bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                        const IFeatureSubSampling& featureSubSampling, const IPruning& pruning,
                        const IPostProcessor& postProcessor, uint32 minCoverage, intp maxConditions,
                        intp maxHeadRefinements, int numThreads, RNG& rng, IModelBuilder& modelBuilder) const override;

};
