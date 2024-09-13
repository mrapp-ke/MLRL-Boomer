#include "mlrl/common/input/feature_vector_binned.hpp"

BinnedFeatureVector::BinnedFeatureVector(float32* thresholds, uint32* indices, uint32* indptr, uint32 numBins,
                                         uint32 numIndices, uint32 sparseBinIndex)
    : CompressedVector(indices, indptr), thresholds(thresholds), numBins(numBins), sparseBinIndex(sparseBinIndex) {}

BinnedFeatureVector::BinnedFeatureVector(const BinnedFeatureVector& other)
    : CompressedVector(other), thresholds(other.thresholds), numBins(other.numBins),
      sparseBinIndex(other.sparseBinIndex) {}

BinnedFeatureVector::BinnedFeatureVector(BinnedFeatureVector&& other)
    : CompressedVector(std::move(other)), thresholds(other.thresholds), numBins(other.numBins),
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

BinnedFeatureVector::threshold_type* BinnedFeatureVector::releaseThresholds() {
    threshold_type* ptr = thresholds;
    thresholds = nullptr;
    return ptr;
}
