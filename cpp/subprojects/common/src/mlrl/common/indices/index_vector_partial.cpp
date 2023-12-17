#include "mlrl/common/indices/index_vector_partial.hpp"

#include "mlrl/common/thresholds/thresholds_subset.hpp"

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

std::unique_ptr<IRuleRefinement> PartialIndexVector::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                          uint32 featureIndex) const {
    return thresholdsSubset.createRuleRefinement(*this, featureIndex);
}
