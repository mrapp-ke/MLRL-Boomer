#include "mlrl/common/input/feature_vector_binned.hpp"

BinnedFeatureVector::BinnedFeatureVector(float32* thresholds, uint32* indices, uint32* indptr, uint32 numBins,
                                         uint32 numIndices, uint32 sparseBinIndex)
    : thresholds(thresholds), indices(indices), indptr(indptr), numBins(numBins), sparseBinIndex(sparseBinIndex) {}

BinnedFeatureVector::BinnedFeatureVector(const BinnedFeatureVector& other)
    : thresholds(other.thresholds), indices(other.indices), indptr(other.indptr), numBins(other.numBins),
      sparseBinIndex(other.sparseBinIndex) {}

BinnedFeatureVector::BinnedFeatureVector(BinnedFeatureVector&& other)
    : thresholds(other.thresholds), indices(other.indices), indptr(other.indptr), numBins(other.numBins),
      sparseBinIndex(other.sparseBinIndex) {}

BinnedFeatureVector::threshold_const_iterator BinnedFeatureVector::thresholds_cbegin() const {
    return thresholds;
}

BinnedFeatureVector::threshold_const_iterator BinnedFeatureVector::thresholds_cend() const {
    return &thresholds[numBins - 1];
}

BinnedFeatureVector::threshold_iterator BinnedFeatureVector::thresholds_begin() {
    return thresholds;
}

BinnedFeatureVector::threshold_iterator BinnedFeatureVector::thresholds_end() {
    return &thresholds[numBins - 1];
}

BinnedFeatureVector::index_const_iterator BinnedFeatureVector::indices_cbegin(uint32 index) const {
    return &indices[indptr[index]];
}

BinnedFeatureVector::index_const_iterator BinnedFeatureVector::indices_cend(uint32 index) const {
    return &indices[indptr[index + 1]];
}

BinnedFeatureVector::index_iterator BinnedFeatureVector::indices_begin(uint32 index) {
    return &indices[indptr[index]];
}

BinnedFeatureVector::index_iterator BinnedFeatureVector::indices_end(uint32 index) {
    return &indices[indptr[index + 1]];
}

BinnedFeatureVector::threshold_type* BinnedFeatureVector::releaseThresholds() {
    threshold_type* ptr = thresholds;
    thresholds = nullptr;
    return ptr;
}

BinnedFeatureVector::index_type* BinnedFeatureVector::releaseIndices() {
    index_type* ptr = indices;
    indices = nullptr;
    return ptr;
}

BinnedFeatureVector::index_type* BinnedFeatureVector::releaseIndptr() {
    index_type* ptr = indptr;
    indptr = nullptr;
    return ptr;
}
