#include "mlrl/common/input/feature_vector_nominal.hpp"

#include <cstdlib>

NominalFeatureVector::NominalFeatureVector(int32* values, uint32* indices, uint32* indptr, uint32 numValues,
                                           uint32 numIndices, int32 majorityValue)
    : values(values), indices(indices), indptr(indptr), numValues(numValues), majorityValue(majorityValue) {}

NominalFeatureVector::NominalFeatureVector(const NominalFeatureVector& other)
    : values(other.values), indices(other.indices), indptr(other.indptr), numValues(other.numValues),
      majorityValue(other.majorityValue) {}

NominalFeatureVector::NominalFeatureVector(NominalFeatureVector&& other)
    : values(other.values), indices(other.indices), indptr(other.indptr), numValues(other.numValues),
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

NominalFeatureVector::index_iterator NominalFeatureVector::indices_begin(uint32 index) {
    return &indices[indptr[index]];
}

NominalFeatureVector::index_iterator NominalFeatureVector::indices_end(uint32 index) {
    return &indices[indptr[index + 1]];
}

NominalFeatureVector::index_const_iterator NominalFeatureVector::indices_cbegin(uint32 index) const {
    return &indices[indptr[index]];
}

NominalFeatureVector::index_const_iterator NominalFeatureVector::indices_cend(uint32 index) const {
    return &indices[indptr[index + 1]];
}

NominalFeatureVector::value_type* NominalFeatureVector::releaseValues() {
    value_type* ptr = values;
    values = nullptr;
    return ptr;
}

NominalFeatureVector::index_type* NominalFeatureVector::releaseIndices() {
    index_type* ptr = indices;
    indices = nullptr;
    return ptr;
}

NominalFeatureVector::index_type* NominalFeatureVector::releaseIndptr() {
    index_type* ptr = indptr;
    indptr = nullptr;
    return ptr;
}
