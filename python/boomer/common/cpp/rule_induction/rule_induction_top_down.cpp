#include "rule_induction_top_down.h"


void TopDownRuleInduction::induceDefaultRule(IStatisticsProvider& statisticsProvider,
                                             const IHeadRefinementFactory* headRefinementFactory,
                                             IModelBuilder& modelBuilder) const {
    // TODO
}

bool TopDownRuleInduction::induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                      const IWeightVector& weights, const IFeatureSubSampling& featureSubSampling,
                                      const IPruning& pruning, const IPostProcessor& postProcessor, uint32 minCoverage,
                                      intp maxConditions, intp maxHeadRefinements, int numThreads, RNG& rng,
                                      IModelBuilder& modelBuilder) const {
    // TODO
    return false;
}
