#include "mlrl/common/input/feature_vector_nominal.hpp"

NominalFeatureVector::NominalFeatureVector(int32* values, uint32* indices, uint32* indptr, uint32 numValues,
                                           uint32 numIndices, int32 majorityValue)
    : CompressedVector(indices, indptr), values(values), numValues(numValues), majorityValue(majorityValue) {}

NominalFeatureVector::NominalFeatureVector(const NominalFeatureVector& other)
    : CompressedVector(other), values(other.values), numValues(other.numValues), majorityValue(other.majorityValue) {}

NominalFeatureVector::NominalFeatureVector(NominalFeatureVector&& other)
    : CompressedVector(std::move(other)), values(other.values), numValues(other.numValues),
      majorityValue(other.majorityValue) {}

NominalFeatureVector::value_iterator NominalFeatureVector::values_begin() {
    return values;
}

NominalFeatureVector::value_iterator NominalFeatureVector::values_end() {
    return &values[numValues];
}

NominalFeatureVector::value_const_iterator NominalFeatureVector::values_cbegin() const {
    return values;
}

NominalFeatureVector::value_const_iterator NominalFeatureVector::values_cend() const {
    return &values[numValues];
}

NominalFeatureVector::value_type* NominalFeatureVector::releaseValues() {
    value_type* ptr = values;
    values = nullptr;
    return ptr;
}
