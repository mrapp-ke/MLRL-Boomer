#include "mlrl/common/rule_refinement/rule_refinement_feature_based.hpp"

#include "mlrl/common/rule_refinement/rule_refinement_search.hpp"

template<typename IndexVector, typename Comparator>
static inline void findRefinementInternally(const IndexVector& labelIndices, uint32 featureIndex,
                                            uint32 numExamplesWithNonZeroWeights, IRuleRefinementCallback& callback,
                                            Comparator& comparator, uint32 minCoverage) {
    // Invoke the callback...
    IRuleRefinementCallback::Result callbackResult = callback.get();
    const IImmutableWeightedStatistics& statistics = callbackResult.statistics;
    const IFeatureVector& featureVector = callbackResult.featureVector;

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices);

    RuleRefinementSearch ruleRefinementSearch;
    Refinement refinement;
    refinement.featureIndex = featureIndex;
    featureVector.searchForRefinement(ruleRefinementSearch, *statisticsSubsetPtr, comparator,
                                      numExamplesWithNonZeroWeights, minCoverage, refinement);
}

template<typename IndexVector>
FeatureBasedRuleRefinement<IndexVector>::FeatureBasedRuleRefinement(
  const IndexVector& labelIndices, uint32 featureIndex, uint32 numExamplesWithNonZeroWeights,
  std::unique_ptr<IRuleRefinementCallback> callbackPtr)
    : labelIndices_(labelIndices), featureIndex_(featureIndex),
      numExamplesWithNonZeroWeights_(numExamplesWithNonZeroWeights), callbackPtr_(std::move(callbackPtr)) {}

template<typename IndexVector>
void FeatureBasedRuleRefinement<IndexVector>::findRefinement(SingleRefinementComparator& comparator,
                                                             uint32 minCoverage) const {
    findRefinementInternally(labelIndices_, featureIndex_, numExamplesWithNonZeroWeights_, *callbackPtr_, comparator,
                             minCoverage);
}

template<typename IndexVector>
void FeatureBasedRuleRefinement<IndexVector>::findRefinement(FixedRefinementComparator& comparator,
                                                             uint32 minCoverage) const {
    findRefinementInternally(labelIndices_, featureIndex_, numExamplesWithNonZeroWeights_, *callbackPtr_, comparator,
                             minCoverage);
}

template class FeatureBasedRuleRefinement<CompleteIndexVector>;
template class FeatureBasedRuleRefinement<PartialIndexVector>;
