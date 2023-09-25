#include "mlrl/common/input/feature_vector_nominal.hpp"

NominalFeatureVector::NominalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue)
    : values_(new int32[numValues]), indices_(new uint32[numExamples]), indptr_(new uint32[numValues + 1]),
      numValues_(numValues), majorityValue_(majorityValue) {
    indptr_[numValues] = numExamples;
}

NominalFeatureVector::~NominalFeatureVector() {
    delete[] values_;
    delete[] indices_;
    delete[] indptr_;
}

NominalFeatureVector::value_iterator NominalFeatureVector::values_begin() {
    return values_;
}

NominalFeatureVector::value_iterator NominalFeatureVector::values_end() {
    return &values_[numValues_];
}

NominalFeatureVector::value_const_iterator NominalFeatureVector::values_cbegin() const {
    return values_;
}

NominalFeatureVector::value_const_iterator NominalFeatureVector::values_cend() const {
    return &values_[numValues_];
}

NominalFeatureVector::index_iterator NominalFeatureVector::indices_begin(uint32 index) {
    return &indices_[indptr_[index]];
}

NominalFeatureVector::index_iterator NominalFeatureVector::indices_end(uint32 index) {
    return &indices_[indptr_[index + 1]];
}

NominalFeatureVector::index_const_iterator NominalFeatureVector::indices_cbegin(uint32 index) const {
    return &indices_[indptr_[index]];
}

NominalFeatureVector::index_const_iterator NominalFeatureVector::indices_cend(uint32 index) const {
    return &indices_[indptr_[index + 1]];
}

NominalFeatureVector::indptr_iterator NominalFeatureVector::indptr_begin() {
    return indptr_;
}

NominalFeatureVector::indptr_iterator NominalFeatureVector::indptr_end() {
    return &indptr_[numValues_];
}

int32 NominalFeatureVector::getMajorityValue() const {
    return majorityValue_;
}

uint32 NominalFeatureVector::getNumElements() const {
    return numValues_;
}
