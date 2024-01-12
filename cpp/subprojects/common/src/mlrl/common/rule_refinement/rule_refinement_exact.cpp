#include "mlrl/common/rule_refinement/rule_refinement_exact.hpp"

#include "mlrl/common/rule_refinement/rule_refinement_search.hpp"

template<typename IndexVector, typename Comparator>
static inline void findRefinementInternally(
  const IndexVector& indexVector, IRuleRefinementCallback<IImmutableWeightedStatistics, IFeatureVector>& callback,
  Comparator& comparator, uint32 minCoverage) {
    IRuleRefinementCallback<IImmutableWeightedStatistics, IFeatureVector>::Result callbackResult = callback.get();
    const IFeatureVector& featureVector = callbackResult.vector;
    RuleRefinementSearch ruleRefinementSearch;
    featureVector.searchForRefinement(ruleRefinementSearch, comparator, minCoverage);
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
