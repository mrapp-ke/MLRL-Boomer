#include "mlrl/common/model/head_partial.hpp"

PartialHead::PartialHead(uint32 numElements)
    : scores_(DenseVector<float64>(numElements)), labelIndices_(DenseVector<uint32>(numElements)) {}

uint32 PartialHead::getNumElements() const {
    return scores_.getNumElements();
}

PartialHead::score_iterator PartialHead::scores_begin() {
    return scores_.begin();
}

PartialHead::score_iterator PartialHead::scores_end() {
    return scores_.end();
}

PartialHead::score_const_iterator PartialHead::scores_cbegin() const {
    return scores_.cbegin();
}

PartialHead::score_const_iterator PartialHead::scores_cend() const {
    return scores_.cend();
}

PartialHead::index_iterator PartialHead::indices_begin() {
    return labelIndices_.begin();
}

PartialHead::index_iterator PartialHead::indices_end() {
    return labelIndices_.end();
}

PartialHead::index_const_iterator PartialHead::indices_cbegin() const {
    return labelIndices_.cbegin();
}

PartialHead::index_const_iterator PartialHead::indices_cend() const {
    return labelIndices_.cend();
}

void PartialHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    partialHeadVisitor(*this);
}
