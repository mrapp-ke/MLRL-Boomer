#include "mlrl/common/model/head_partial.hpp"

PartialHead::PartialHead(uint32 numElements) : vector_(SparseArraysVector<float64>(numElements)) {}

uint32 PartialHead::getNumElements() const {
    return vector_.getNumElements();
}

PartialHead::value_iterator PartialHead::values_begin() {
    return vector_.values_begin();
}

PartialHead::value_iterator PartialHead::values_end() {
    return vector_.values_end();
}

PartialHead::value_const_iterator PartialHead::values_cbegin() const {
    return vector_.values_cbegin();
}

PartialHead::value_const_iterator PartialHead::values_cend() const {
    return vector_.values_cend();
}

PartialHead::index_iterator PartialHead::indices_begin() {
    return vector_.indices_begin();
}

PartialHead::index_iterator PartialHead::indices_end() {
    return vector_.indices_end();
}

PartialHead::index_const_iterator PartialHead::indices_cbegin() const {
    return vector_.indices_cbegin();
}

PartialHead::index_const_iterator PartialHead::indices_cend() const {
    return vector_.indices_cend();
}

void PartialHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    partialHeadVisitor(*this);
}
