#include "rule_induction_top_down.h"
#include "../indices/index_vector_full.h"


void TopDownRuleInduction::induceDefaultRule(IStatisticsProvider& statisticsProvider,
                                             const IHeadRefinementFactory* headRefinementFactory,
                                             IModelBuilder& modelBuilder) const {
    if (headRefinementFactory != nullptr) {
        IStatistics& statistics = statisticsProvider.get();
        uint32 numStatistics = statistics.getNumStatistics();
        uint32 numLabels = statistics.getNumLabels();
        statistics.resetSampledStatistics();

        for (uint32 i = 0; i < numStatistics; i++) {
            statistics.addSampledStatistic(i, 1);
        }

        FullIndexVector labelIndices(numLabels);
        std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices.createSubset(statistics);
        std::unique_ptr<IHeadRefinement> headRefinementPtr = headRefinementFactory->create(labelIndices);
        headRefinementPtr->findHead(nullptr, *statisticsSubsetPtr, true, false);
        std::unique_ptr<AbstractEvaluatedPrediction> defaultPredictionPtr = headRefinementPtr->pollHead();
        statisticsProvider.switchRuleEvaluation();

        for (uint32 i = 0; i < numStatistics; i++) {
            defaultPredictionPtr->apply(statistics, i);
        }

        modelBuilder.setDefaultRule(*defaultPredictionPtr);
    } else {
        statisticsProvider.switchRuleEvaluation();
    }
}

bool TopDownRuleInduction::induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                      const IWeightVector& weights, const IFeatureSubSampling& featureSubSampling,
                                      const IPruning& pruning, const IPostProcessor& postProcessor, uint32 minCoverage,
                                      intp maxConditions, intp maxHeadRefinements, int numThreads, RNG& rng,
                                      IModelBuilder& modelBuilder) const {
    // TODO
    return false;
}
