#include "mlrl/common/model/head_complete.hpp"

static inline void visitInternally(const CompleteHead<float32>& head,
                                   IHead::CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                                   IHead::CompleteHeadVisitor<float64> complete64BitHeadVisitor) {
    complete32BitHeadVisitor(head);
}

static inline void visitInternally(const CompleteHead<float64>& head,
                                   IHead::CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                                   IHead::CompleteHeadVisitor<float64> complete64BitHeadVisitor) {
    complete64BitHeadVisitor(head);
}

template<typename ScoreType>
CompleteHead<ScoreType>::CompleteHead(uint32 numElements)
    : VectorDecorator<AllocatedVector<ScoreType>>(AllocatedVector<ScoreType>(numElements)) {}

template<typename ScoreType>
typename CompleteHead<ScoreType>::value_iterator CompleteHead<ScoreType>::values_begin() {
    return this->view.begin();
}

template<typename ScoreType>
typename CompleteHead<ScoreType>::value_iterator CompleteHead<ScoreType>::values_end() {
    return this->view.end();
}

template<typename ScoreType>
typename CompleteHead<ScoreType>::value_const_iterator CompleteHead<ScoreType>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType>
typename CompleteHead<ScoreType>::value_const_iterator CompleteHead<ScoreType>::values_cend() const {
    return this->view.cend();
}

template<typename ScoreType>
void CompleteHead<ScoreType>::visit(CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                                    CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                                    PartialHeadVisitor<float32> partial32BitHeadVisitor,
                                    PartialHeadVisitor<float64> partial64BitHeadVisitor) const {
    visitInternally(*this, complete32BitHeadVisitor, complete64BitHeadVisitor);
}

template class CompleteHead<float32>;
template class CompleteHead<float64>;
