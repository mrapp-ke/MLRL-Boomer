#include "mlrl/common/binning/bin_weight_vector.hpp"

BinWeightVector::BinWeightVector(uint32 numElements)
    : ClearableViewDecorator<VectorDecorator<AllocatedVector<uint32>>>(AllocatedVector<uint32>(numElements)) {}

uint32 BinWeightVector::getNumNonZeroWeights() const {
    throw std::runtime_error("Function BinWeightVector::getNumNonZeroWeights is not implemented");
}

void BinWeightVector::increaseWeight(uint32 pos) {
    this->view.array[pos] += 1;
}

bool BinWeightVector::operator[](uint32 pos) const {
    return this->view.array[pos] != 0;
}
