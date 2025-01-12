#include "mlrl/common/model/head_partial.hpp"

PartialHead::PartialHead(uint32 numElements)
    : IterableIndexedVectorDecorator<IndexedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<float64>>>(
        CompositeVector<AllocatedVector<uint32>, AllocatedVector<float64>>(AllocatedVector<uint32>(numElements),
                                                                           AllocatedVector<float64>(numElements))) {}

void PartialHead::visit(std::optional<CompleteHeadVisitor> completeHeadVisitor,
                        std::optional<PartialHeadVisitor> partialHeadVisitor) const {
    if (partialHeadVisitor) {
        (*partialHeadVisitor)(*this);
    }
}
