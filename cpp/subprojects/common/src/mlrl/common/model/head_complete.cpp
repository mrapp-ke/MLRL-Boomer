#include "mlrl/common/model/head_complete.hpp"

CompleteHead::CompleteHead(uint32 numElements) : vector_(DenseVector<float64>(numElements)) {}

uint32 CompleteHead::getNumElements() const {
    return vector_.getNumElements();
}

CompleteHead::score_iterator CompleteHead::scores_begin() {
    return vector_.begin();
}

CompleteHead::score_iterator CompleteHead::scores_end() {
    return vector_.end();
}

CompleteHead::score_const_iterator CompleteHead::scores_cbegin() const {
    return vector_.cbegin();
}

CompleteHead::score_const_iterator CompleteHead::scores_cend() const {
    return vector_.cend();
}

void CompleteHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    completeHeadVisitor(*this);
}
