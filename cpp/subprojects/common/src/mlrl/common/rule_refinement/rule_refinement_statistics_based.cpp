#include "mlrl/common/rule_refinement/rule_refinement_statistics_based.hpp"

#include "mlrl/common/rule_refinement/feature_based_search.hpp"
#include "mlrl/common/rule_refinement/feature_subspace.hpp"
#include "mlrl/common/util/openmp.hpp"

template<typename RefinementComparator>
struct RuleRefinementEntry final {
    public:

        std::unique_ptr<RefinementComparator> comparatorPtr;

        std::unique_ptr<IFeatureSubspace::ICallback> callbackPtr;
};

template<typename RefinementComparator, typename IndexVector>
static inline void findRefinementInternally(RefinementComparator& refinementComparator,
                                            const IndexVector& outputIndices, uint32 featureIndex,
                                            const IWeightedStatistics& statistics, const IFeatureVector& featureVector,
                                            uint32 numExamplesWithNonZeroWeights, uint32 minCoverage) {
    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(outputIndices);

    FeatureBasedSearch featureBasedSearch;
    Refinement refinement;
    refinement.featureIndex = featureIndex;
    featureVector.searchForRefinement(featureBasedSearch, *statisticsSubsetPtr, refinementComparator,
                                      numExamplesWithNonZeroWeights, minCoverage, refinement);
}

template<typename RefinementComparator>
static inline bool findRefinementInternally(RefinementComparator& refinementComparator,
                                            IFeatureSubspace& featureSubspace, const IIndexVector& featureIndices,
                                            const IIndexVector& outputIndices, uint32 minCoverage,
                                            MultiThreadingSettings multiThreadingSettings) {
    bool foundRefinement = false;

    // For each feature, create an object of type `RuleRefinement<RefinementComparator>`...
    uint32 numFeatures = featureIndices.getNumElements();
    RuleRefinementEntry<RefinementComparator>* ruleRefinementEntries =
      new RuleRefinementEntry<RefinementComparator>[numFeatures];

    for (uint32 i = 0; i < numFeatures; i++) {
        uint32 featureIndex = featureIndices.getIndex(i);
        RuleRefinementEntry<RefinementComparator>& ruleRefinementEntry = ruleRefinementEntries[i];
        ruleRefinementEntry.comparatorPtr = std::make_unique<RefinementComparator>(refinementComparator);
        ruleRefinementEntry.callbackPtr = featureSubspace.createCallback(featureIndex);
    }

    // Search for the best condition among all available features to be added to the current rule...
#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numFeatures) firstprivate(ruleRefinementEntries) firstprivate(minCoverage) \
      schedule(dynamic) num_threads(multiThreadingSettings.numThreads)
#endif
    for (int64 i = 0; i < numFeatures; i++) {
        uint32 featureIndex = featureIndices.getIndex(i);
        RuleRefinementEntry<RefinementComparator>& ruleRefinementEntry = ruleRefinementEntries[i];
        RefinementComparator& refinementComparator = *ruleRefinementEntry.comparatorPtr;
        IFeatureSubspace::ICallback::Result callbackResult = ruleRefinementEntry.callbackPtr->get();
        const IFeatureVector& featureVector = callbackResult.featureVector;
        const IWeightedStatistics& statistics = callbackResult.statistics;

        auto partialIndexVectorVisitor = [&](const PartialIndexVector& partialIndexVector) {
            findRefinementInternally(refinementComparator, partialIndexVector, featureIndex, statistics, featureVector,
                                     featureSubspace.getNumCovered(), minCoverage);
        };
        auto completeIndexVectorVisitor = [&](const CompleteIndexVector& completeIndexVector) {
            findRefinementInternally(refinementComparator, completeIndexVector, featureIndex, statistics, featureVector,
                                     featureSubspace.getNumCovered(), minCoverage);
        };
        outputIndices.visit(partialIndexVectorVisitor, completeIndexVectorVisitor);
    }

    // Pick the best refinement among the refinements that have been found for the different features...
    for (uint32 i = 0; i < numFeatures; i++) {
        RuleRefinementEntry<RefinementComparator>& ruleRefinementEntry = ruleRefinementEntries[i];
        foundRefinement |= refinementComparator.merge(*ruleRefinementEntry.comparatorPtr);
    }

    delete[] ruleRefinementEntries;
    return foundRefinement;
}

class StatisticsBasedRuleRefinement final : public IRuleRefinement {
    private:

        const MultiThreadingSettings multiThreadingSettings_;

    public:

        StatisticsBasedRuleRefinement(MultiThreadingSettings multiThreadingSettings)
            : multiThreadingSettings_(multiThreadingSettings) {}

        bool findRefinement(SingleRefinementComparator& comparator, IFeatureSubspace& featureSubspace,
                            const IIndexVector& featureIndices, const IIndexVector& outputIndices,
                            uint32 minCoverage) const override {
            return findRefinementInternally(comparator, featureSubspace, featureIndices, outputIndices, minCoverage,
                                            multiThreadingSettings_);
        }

        bool findRefinement(FixedRefinementComparator& comparator, IFeatureSubspace& featureSubspace,
                            const IIndexVector& featureIndices, const IIndexVector& outputIndices,
                            uint32 minCoverage) const override {
            return findRefinementInternally(comparator, featureSubspace, featureIndices, outputIndices, minCoverage,
                                            multiThreadingSettings_);
        }
};

class StatisticsBasedRuleRefinementFactory final : public IRuleRefinementFactory {
    private:

        const MultiThreadingSettings multiThreadingSettings_;

    public:

        StatisticsBasedRuleRefinementFactory(MultiThreadingSettings multiThreadingSettings)
            : multiThreadingSettings_(multiThreadingSettings) {}

        std::unique_ptr<IRuleRefinement> create() const override {
            return std::make_unique<StatisticsBasedRuleRefinement>(multiThreadingSettings_);
        }
};

StatisticsBasedRuleRefinementConfig::StatisticsBasedRuleRefinementConfig(
  ReadableProperty<IMultiThreadingConfig> multiThreadingConfig)
    : multiThreadingConfig_(multiThreadingConfig) {}

std::unique_ptr<IRuleRefinementFactory> StatisticsBasedRuleRefinementConfig::createRuleRefinementFactory(
  const IFeatureMatrix& featureMatrix, uint32 numOutputs) const {
    MultiThreadingSettings multiThreadingSettings = multiThreadingConfig_.get().getSettings(featureMatrix, numOutputs);
    return std::make_unique<StatisticsBasedRuleRefinementFactory>(multiThreadingSettings);
}
