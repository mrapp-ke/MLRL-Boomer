#include "mlrl/common/model/head_complete.hpp"

CompleteHead::CompleteHead(uint32 numElements)
    : VectorDecorator<AllocatedVector<float64>>(AllocatedVector<float64>(numElements)) {}

CompleteHead::value_iterator CompleteHead::values_begin() {
    return this->view_.array;
}

CompleteHead::value_iterator CompleteHead::values_end() {
    return &this->view_.array[this->view_.numElements];
}

CompleteHead::value_const_iterator CompleteHead::values_cbegin() const {
    return this->view_.array;
}

CompleteHead::value_const_iterator CompleteHead::values_cend() const {
    return &this->view_.array[this->view_.numElements];
}

void CompleteHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    completeHeadVisitor(*this);
}
