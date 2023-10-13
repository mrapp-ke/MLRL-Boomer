#include "mlrl/common/input/feature_vector_nominal.hpp"

#include <cstdlib>

NominalFeatureVector::NominalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue)
    : values_((int32*) malloc(numValues * sizeof(int32))), numValues_(numValues), majorityValue_(majorityValue),
      indices_((uint32*) malloc(numExamples * sizeof(uint32))),
      indptr_((uint32*) malloc((numValues + 1) * sizeof(uint32))) {
    indptr_[numValues] = numExamples;
}

NominalFeatureVector::~NominalFeatureVector() {
    free(values_);
    free(indices_);
    free(indptr_);
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

std::unique_ptr<IFeatureVector> NominalFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureVector> NominalFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    // TODO Implement
    return nullptr;
}
