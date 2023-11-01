#include "mlrl/common/binning/bin_weight_vector.hpp"

BinWeightVector::BinWeightVector(uint32 numElements)
    : ClearableVectorDecorator<VectorDecorator<AllocatedVector<uint32>>>(AllocatedVector<uint32>(numElements)) {}

void BinWeightVector::increaseWeight(uint32 pos) {
    this->view_.array[pos] += 1;
}

bool BinWeightVector::operator[](uint32 pos) const {
    return this->view_.array[pos] != 0;
}
