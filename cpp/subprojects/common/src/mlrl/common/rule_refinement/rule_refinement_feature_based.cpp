#include "mlrl/common/rule_refinement/rule_refinement_feature_based.hpp"

#include "mlrl/common/rule_refinement/feature_based_search.hpp"

template<typename IndexVector, typename Comparator>
static inline void findRefinementInternally(const IndexVector& outputIndices, uint32 featureIndex,
                                            const IWeightedStatistics& statistics, const IFeatureVector& featureVector,
                                            uint32 numExamplesWithNonZeroWeights, Comparator& comparator,
                                            uint32 minCoverage) {
    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(outputIndices);

    FeatureBasedSearch featureBasedSearch;
    Refinement refinement;
    refinement.featureIndex = featureIndex;
    featureVector.searchForRefinement(featureBasedSearch, *statisticsSubsetPtr, comparator,
                                      numExamplesWithNonZeroWeights, minCoverage, refinement);
}

template<typename IndexVector>
FeatureBasedRuleRefinement<IndexVector>::FeatureBasedRuleRefinement(const IndexVector& outputIndices,
                                                                    uint32 featureIndex,
                                                                    const IWeightedStatistics& statistics,
                                                                    const IFeatureVector& featureVector,
                                                                    uint32 numExamplesWithNonZeroWeights)
    : outputIndices_(outputIndices), featureIndex_(featureIndex), statistics_(statistics),
      featureVector_(featureVector), numExamplesWithNonZeroWeights_(numExamplesWithNonZeroWeights) {}

template<typename IndexVector>
void FeatureBasedRuleRefinement<IndexVector>::findRefinement(SingleRefinementComparator& comparator,
                                                             uint32 minCoverage) const {
    findRefinementInternally(outputIndices_, featureIndex_, statistics_, featureVector_, numExamplesWithNonZeroWeights_,
                             comparator, minCoverage);
}

template<typename IndexVector>
void FeatureBasedRuleRefinement<IndexVector>::findRefinement(FixedRefinementComparator& comparator,
                                                             uint32 minCoverage) const {
    findRefinementInternally(outputIndices_, featureIndex_, statistics_, featureVector_, numExamplesWithNonZeroWeights_,
                             comparator, minCoverage);
}

template class FeatureBasedRuleRefinement<CompleteIndexVector>;
template class FeatureBasedRuleRefinement<PartialIndexVector>;
