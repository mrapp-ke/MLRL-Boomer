#include "mlrl/common/indices/index_vector_partial.hpp"

#include "mlrl/common/rule_refinement/feature_subspace.hpp"

PartialIndexVector::PartialIndexVector(uint32 numElements, bool init)
    : ResizableVectorDecorator<DenseVectorDecorator<ResizableVector<uint32>>>(
        ResizableVector<uint32>(numElements, init)) {}

uint32 PartialIndexVector::getNumElements() const {
    return VectorDecorator<ResizableVector<uint32>>::getNumElements();
}

bool PartialIndexVector::isPartial() const {
    return true;
}

uint32 PartialIndexVector::getIndex(uint32 pos) const {
    return (*this)[pos];
}

void PartialIndexVector::visit(PartialIndexVectorVisitor partialIndexVectorVisitor,
                               CompleteIndexVectorVisitor completeIndexVectorVisitor) const {
    partialIndexVectorVisitor(*this);
}
