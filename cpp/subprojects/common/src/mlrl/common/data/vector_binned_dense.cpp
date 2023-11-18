#include "mlrl/common/data/vector_binned_dense.hpp"

template<typename T>
DenseBinnedVector<T>::DenseBinnedVector(uint32 numElements, uint32 numBins)
    : WritableBinnedVectorDecorator<AllocatedVector<uint32>, AllocatedVector<T>>(AllocatedVector<uint32>(numElements),
                                                                                 AllocatedVector<T>(numBins)),
      maxCapacity_(numBins) {}

template<typename T>
void DenseBinnedVector<T>::setNumBins(uint32 numBins, bool freeMemory) {
    if (numBins < maxCapacity_) {
        if (freeMemory) {
            this->secondView_.array = reallocateMemory(this->secondView_.array, numBins);
            maxCapacity_ = numBins;
        }
    } else if (numBins > maxCapacity_) {
        this->secondView_.array = reallocateMemory(this->secondView_.array, numBins);
        maxCapacity_ = numBins;
    }

    this->secondView_.numElements = numBins;
}

template class DenseBinnedVector<uint8>;
template class DenseBinnedVector<uint32>;
template class DenseBinnedVector<int64>;
template class DenseBinnedVector<float32>;
template class DenseBinnedVector<float64>;
