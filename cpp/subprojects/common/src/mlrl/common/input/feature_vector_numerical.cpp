#include "mlrl/common/input/feature_vector_numerical.hpp"

NumericalFeatureVector::NumericalFeatureVector(IndexedValue<float32>* array, uint32 numElements)
    : Vector<IndexedValue<float32>>(array, numElements), sparseValue(0) {}

NumericalFeatureVector::NumericalFeatureVector(const NumericalFeatureVector& other)
    : Vector<IndexedValue<float32>>(other), sparseValue(other.sparseValue) {}

NumericalFeatureVector::NumericalFeatureVector(NumericalFeatureVector&& other)
    : Vector<IndexedValue<float32>>(std::move(other)), sparseValue(other.sparseValue) {}
