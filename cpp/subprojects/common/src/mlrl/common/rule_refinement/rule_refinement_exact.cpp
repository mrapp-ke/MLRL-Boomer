#include "mlrl/common/rule_refinement/rule_refinement_exact.hpp"

#include "mlrl/common/rule_refinement/rule_refinement_search.hpp"

template<typename IndexVector, typename Comparator>
static inline void findRefinementInternally(
  const IndexVector& labelIndices, IRuleRefinementCallback<IImmutableWeightedStatistics, IFeatureVector>& callback,
  Comparator& comparator, uint32 minCoverage) {
    // Invoke the callback...
    IRuleRefinementCallback<IImmutableWeightedStatistics, IFeatureVector>::Result callbackResult = callback.get();
    const IImmutableWeightedStatistics& statistics = callbackResult.statistics;
    const IFeatureVector& featureVector = callbackResult.vector;

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices);

    RuleRefinementSearch ruleRefinementSearch;
    featureVector.searchForRefinement(ruleRefinementSearch, *statisticsSubsetPtr, comparator, minCoverage);
}

template<typename IndexVector>
ExactRuleRefinement<IndexVector>::ExactRuleRefinement(const IndexVector& labelIndices,
                                                      std::unique_ptr<Callback> callbackPtr)
    : labelIndices_(labelIndices), callbackPtr_(std::move(callbackPtr)) {}

template<typename IndexVector>
void ExactRuleRefinement<IndexVector>::findRefinement(SingleRefinementComparator& comparator, uint32 minCoverage) {
    findRefinementInternally(labelIndices_, *callbackPtr_, comparator, minCoverage);
}

template<typename IndexVector>
void ExactRuleRefinement<IndexVector>::findRefinement(FixedRefinementComparator& comparator, uint32 minCoverage) {
    findRefinementInternally(labelIndices_, *callbackPtr_, comparator, minCoverage);
}

template class ExactRuleRefinement<CompleteIndexVector>;
template class ExactRuleRefinement<PartialIndexVector>;
