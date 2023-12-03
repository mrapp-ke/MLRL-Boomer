#include "mlrl/common/input/feature_vector.hpp"

#include <algorithm>

FeatureVector::FeatureVector(uint32 numElements)
    : ResizableVectorDecorator<DenseVectorDecorator<ResizableVector<IndexedValue<float32>>>>(
      ResizableVector<IndexedValue<float32>>(numElements)) {}

void FeatureVector::sortByValues() {
    std::sort(this->begin(), this->end(), IndexedValue<float32>::CompareValue());
}
