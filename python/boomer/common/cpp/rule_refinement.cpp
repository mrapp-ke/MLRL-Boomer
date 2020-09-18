#include "rule_refinement.h"


AbstractRuleRefinement::~AbstractRuleRefinement() {

}

Refinement AbstractRuleRefinement::findRefinement(AbstractHeadRefinement* headRefinement) {
    Refinement refinement;
    return refinement;
}

RuleRefinementImpl::RuleRefinementImpl(AbstractStatistics* statistics, IndexedFloat32ArrayWrapper* indexedArrayWrapper,
                                       const uint32* weights, uint32 totalSumOfWeights, uint32 featureIndex,
                                       bool nominal) {
    statistics_ = statistics;
    indexedArrayWrapper_ = indexedArrayWrapper;
    weights_ = weights;
    totalSumOfWeights_ = totalSumOfWeights;
    featureIndex_ = featureIndex;
    nominal_ = nominal;
}

RuleRefinementImpl::~RuleRefinementImpl() {

}

Refinement RuleRefinementImpl::findRefinement(AbstractHeadRefinement* headRefinement) {
    Refinement refinement;
    // TODO Implement
    return refinement;
}
