#include "mlrl/common/model/head_complete.hpp"

CompleteHead::CompleteHead(uint32 numElements)
    : VectorDecorator<AllocatedVector<float64>>(AllocatedVector<float64>(numElements)) {}

CompleteHead::value_iterator CompleteHead::values_begin() {
    return this->view_.begin();
}

CompleteHead::value_iterator CompleteHead::values_end() {
    return this->view_.end();
}

CompleteHead::value_const_iterator CompleteHead::values_cbegin() const {
    return this->view_.cbegin();
}

CompleteHead::value_const_iterator CompleteHead::values_cend() const {
    return this->view_.cend();
}

void CompleteHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    completeHeadVisitor(*this);
}
