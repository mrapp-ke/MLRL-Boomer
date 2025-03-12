#include "mlrl/common/model/head_partial.hpp"

template<typename ScoreType>
PartialHead<ScoreType>::PartialHead(uint32 numElements)
    : IterableIndexedVectorDecorator<IndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<ScoreType>>>(
        CompositeVector<AllocatedVector<uint32>, AllocatedVector<ScoreType>>(
          AllocatedVector<uint32>(numElements), AllocatedVector<ScoreType>(numElements))) {}

template<typename ScoreType>
void PartialHead<ScoreType>::visit(CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                                   PartialHeadVisitor<float64> partial64BitHeadVisitor) const {
    partial64BitHeadVisitor(*this);
}

template class PartialHead<float64>;
