#include "mlrl/common/model/head_complete.hpp"

CompleteHead::CompleteHead(uint32 numElements) : vector_(numElements) {}

uint32 CompleteHead::getNumElements() const {
    return vector_.getNumElements();
}

CompleteHead::value_iterator CompleteHead::values_begin() {
    return vector_.begin();
}

CompleteHead::value_iterator CompleteHead::values_end() {
    return vector_.end();
}

CompleteHead::value_const_iterator CompleteHead::values_cbegin() const {
    return vector_.cbegin();
}

CompleteHead::value_const_iterator CompleteHead::values_cend() const {
    return vector_.cend();
}

void CompleteHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    completeHeadVisitor(*this);
}
