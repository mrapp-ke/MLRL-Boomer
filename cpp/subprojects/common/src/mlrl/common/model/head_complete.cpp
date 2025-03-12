#include "mlrl/common/model/head_complete.hpp"

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
void CompleteHead<ScoreType>::visit(CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                                    PartialHeadVisitor<float64> partial64BitHeadVisitor) const {
    complete64BitHeadVisitor(*this);
}

template class CompleteHead<float64>;
