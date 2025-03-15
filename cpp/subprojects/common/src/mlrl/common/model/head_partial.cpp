#include "mlrl/common/model/head_partial.hpp"

static inline void visitInternally(const PartialHead<float32>& head,
                                   IHead::PartialHeadVisitor<float32> partial32BitHeadVisitor,
                                   IHead::PartialHeadVisitor<float64> partial64BitHeadVisitor) {
    partial32BitHeadVisitor(head);
}

static inline void visitInternally(const PartialHead<float64>& head,
                                   IHead::PartialHeadVisitor<float32> partial32BitHeadVisitor,
                                   IHead::PartialHeadVisitor<float64> partial64BitHeadVisitor) {
    partial64BitHeadVisitor(head);
}

template<typename ScoreType>
PartialHead<ScoreType>::PartialHead(uint32 numElements)
    : IterableIndexedVectorDecorator<IndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<ScoreType>>>(
        CompositeVector<AllocatedVector<uint32>, AllocatedVector<ScoreType>>(
          AllocatedVector<uint32>(numElements), AllocatedVector<ScoreType>(numElements))) {}

template<typename ScoreType>
void PartialHead<ScoreType>::visit(CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                                   CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                                   PartialHeadVisitor<float32> partial32BitHeadVisitor,
                                   PartialHeadVisitor<float64> partial64BitHeadVisitor) const {
    visitInternally(*this, partial32BitHeadVisitor, partial64BitHeadVisitor);
}

template class PartialHead<float32>;
template class PartialHead<float64>;
