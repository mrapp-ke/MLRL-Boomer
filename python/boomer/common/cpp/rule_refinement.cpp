#include "rule_refinement.h"
#include <memory>

AbstractRuleRefinement::~AbstractRuleRefinement() {

}

Refinement AbstractRuleRefinement::findRefinement(AbstractHeadRefinement* headRefinement,
                                                  PredictionCandidate* currentHead, uint32 numLabelIndices,
                                                  const uint32* labelIndices) {
    Refinement refinement;
    return refinement;
}

RuleRefinementImpl::RuleRefinementImpl(AbstractStatistics* statistics, IndexedFloat32Array* indexedArray,
                                       const uint32* weights, uint32 totalSumOfWeights, uint32 featureIndex,
                                       bool nominal) {
    statistics_ = statistics;
    indexedArray_ = indexedArray;
    weights_ = weights;
    totalSumOfWeights_ = totalSumOfWeights;
    featureIndex_ = featureIndex;
    nominal_ = nominal;
}

RuleRefinementImpl::~RuleRefinementImpl() {

}

Refinement RuleRefinementImpl::findRefinement(AbstractHeadRefinement* headRefinement,
                                              PredictionCandidate* currentHead, uint32 numLabelIndices,
                                              const uint32* labelIndices) {
    // The current refinement of the existing rule
    Refinement refinement;
    refinement.featureIndex = featureIndex_;
    refinement.head = NULL;
    // The best head seen so far
    PredictionCandidate* bestHead = currentHead;
    // The `AbstractRefinementSearch` to be used for evaluating refinements
    std::unique_ptr<AbstractRefinementSearch> refinementSearchPtr;
    refinementSearchPtr.reset(statistics_->beginSearch(numLabelIndices, labelIndices));
    // The example indices and feature values to be iterated
    IndexedFloat32* indexedValues = indexedArray_->data;
    uint32 numIndexedValues = indexedArray_->numElements;

    // TODO Implement

    return refinement;
}
