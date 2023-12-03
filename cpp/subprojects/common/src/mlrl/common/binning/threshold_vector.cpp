#include "mlrl/common/binning/threshold_vector.hpp"

ThresholdVector::ThresholdVector(MissingFeatureVector& missingFeatureVector, uint32 numElements, bool init)
    : ResizableVectorDecorator<DenseVectorDecorator<ResizableVector<float32>>>(
      ResizableVector<float32>(numElements, init)),
      MissingFeatureVector(missingFeatureVector), sparseBinIndex_(numElements) {}

uint32 ThresholdVector::getSparseBinIndex() const {
    return sparseBinIndex_;
}

void ThresholdVector::setSparseBinIndex(uint32 sparseBinIndex) {
    uint32 numElements = this->getNumElements();

    if (sparseBinIndex > numElements) {
        sparseBinIndex_ = numElements;
    } else {
        sparseBinIndex_ = sparseBinIndex;
    }
}

void ThresholdVector::setNumElements(uint32 numElements, bool freeMemory) {
    ResizableVectorDecorator<DenseVectorDecorator<ResizableVector<float32>>>::setNumElements(numElements, freeMemory);

    if (sparseBinIndex_ > numElements) {
        sparseBinIndex_ = numElements;
    }
}
