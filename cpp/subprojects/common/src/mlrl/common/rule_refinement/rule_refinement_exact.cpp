#include "mlrl/common/rule_refinement/rule_refinement_exact.hpp"

template<typename IndexVector>
ExactRuleRefinement<IndexVector>::ExactRuleRefinement(const IndexVector& labelIndices,
                                                      std::unique_ptr<Callback> callbackPtr)
    : labelIndices_(labelIndices), callbackPtr_(std::move(callbackPtr)) {}

template<typename IndexVector>
void ExactRuleRefinement<IndexVector>::findRefinement(SingleRefinementComparator& comparator, uint32 minCoverage) {
    // TODO Implement
    /*
    findRefinementInternally(labelIndices_, numExamples_, featureIndex_, ordinal_, nominal_, minCoverage,
                             hasZeroWeights_, *callbackPtr_, comparator);
    */
}

template<typename IndexVector>
void ExactRuleRefinement<IndexVector>::findRefinement(FixedRefinementComparator& comparator, uint32 minCoverage) {
    // TODO Implement
    /*
    findRefinementInternally(labelIndices_, numExamples_, featureIndex_, ordinal_, nominal_, minCoverage,
                             hasZeroWeights_, *callbackPtr_, comparator);
    */
}

template class ExactRuleRefinement<CompleteIndexVector>;
template class ExactRuleRefinement<PartialIndexVector>;
