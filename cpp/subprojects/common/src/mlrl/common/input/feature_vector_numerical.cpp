#include "mlrl/common/input/feature_vector_numerical.hpp"

NumericalFeatureVector::NumericalFeatureVector(IndexedValue<float32>* array, uint32 numElements, uint32 padding,
                                               float32 sparseValue, bool sparse)
    : Vector<IndexedValue<float32>>(array, numElements, padding), sparseValue(sparseValue), sparse(sparse) {}

NumericalFeatureVector::NumericalFeatureVector(const NumericalFeatureVector& other)
    : Vector<IndexedValue<float32>>(other), sparseValue(other.sparseValue), sparse(other.sparse) {}

NumericalFeatureVector::NumericalFeatureVector(NumericalFeatureVector&& other)
    : Vector<IndexedValue<float32>>(std::move(other)), sparseValue(other.sparseValue), sparse(other.sparse) {}
