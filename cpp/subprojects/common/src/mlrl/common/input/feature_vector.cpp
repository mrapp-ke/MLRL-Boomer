#include "mlrl/common/input/feature_vector.hpp"

#include <algorithm>

FeatureVector::FeatureVector(uint32 numElements)
    : ResizableVectorDecorator<WritableVectorDecorator<AllocatedVector<IndexedValue<float32>>>>(
      AllocatedVector<IndexedValue<float32>>(numElements)) {}

void FeatureVector::sortByValues() {
    std::sort(this->begin(), this->end(), IndexedValue<float32>::CompareValue());
}
