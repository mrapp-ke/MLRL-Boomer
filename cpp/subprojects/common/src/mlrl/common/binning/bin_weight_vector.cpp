#include "mlrl/common/binning/bin_weight_vector.hpp"

#include "mlrl/common/util/view_functions.hpp"

BinWeightVector::BinWeightVector(uint32 numElements)
    : VectorDecorator<AllocatedView<Vector<uint32>>>(AllocatedView<Vector<uint32>>(numElements)) {}

void BinWeightVector::clear() {
    setViewToZeros(this->view_.array, this->view_.numElements);
}

void BinWeightVector::increaseWeight(uint32 pos) {
    this->view_.array[pos] += 1;
}

bool BinWeightVector::operator[](uint32 pos) const {
    return this->view_.array[pos] != 0;
}
