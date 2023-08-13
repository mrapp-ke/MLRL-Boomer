#include "mlrl/common/model/head_complete.hpp"

CompleteHead::CompleteHead(uint32 numElements) : scores_(DenseVector<float64>(numElements)) {}

uint32 CompleteHead::getNumElements() const {
    return scores_.getNumElements();
}

CompleteHead::score_iterator CompleteHead::scores_begin() {
    return scores_.begin();
}

CompleteHead::score_iterator CompleteHead::scores_end() {
    return scores_.end();
}

CompleteHead::score_const_iterator CompleteHead::scores_cbegin() const {
    return scores_.cbegin();
}

CompleteHead::score_const_iterator CompleteHead::scores_cend() const {
    return scores_.cend();
}

void CompleteHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    completeHeadVisitor(*this);
}
