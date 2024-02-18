#include "mlrl/common/binning/bin_index_vector_dense.hpp"

DenseBinIndexVector::DenseBinIndexVector(uint32 numElements)
    : DenseVectorDecorator<AllocatedVector<uint32>>(AllocatedVector<uint32>(numElements)) {}

uint32 DenseBinIndexVector::getBinIndex(uint32 exampleIndex) const {
    return (*this)[exampleIndex];
}

void DenseBinIndexVector::setBinIndex(uint32 exampleIndex, uint32 binIndex) {
    (*this)[exampleIndex] = binIndex;
}
