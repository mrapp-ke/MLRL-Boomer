#include "rule_refinement.h"


AbstractRuleRefinement::~AbstractRuleRefinement() {

}

Refinement AbstractRuleRefinement::findRefinement(AbstractHeadRefinement* headRefinement,
                                                  PredictionCandidate* currentHead) {
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

Refinement RuleRefinementImpl::findRefinement(AbstractHeadRefinement* headRefinement,
                                              PredictionCandidate* currentHead) {
    // The current refinement of the existing rule
    Refinement refinement;
    refinement.featureIndex = featureIndex_;
    refinement.head = NULL;
    // The best head seen so far
    PredictionCandidate* bestHead = currentHead;

    // TODO Implement

    return refinement;
}
