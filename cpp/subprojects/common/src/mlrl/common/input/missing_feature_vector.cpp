#include "mlrl/common/input/missing_feature_vector.hpp"

MissingFeatureVector::MissingFeatureVector()
    : missingIndicesPtr_(
      std::make_unique<ClearableViewDecorator<ViewDecorator<AllocatedBinaryDokVector>>>(AllocatedBinaryDokVector())) {}

MissingFeatureVector::MissingFeatureVector(MissingFeatureVector& missingFeatureVector)
    : missingIndicesPtr_(std::move(missingFeatureVector.missingIndicesPtr_)) {}

MissingFeatureVector::missing_index_const_iterator MissingFeatureVector::missing_indices_cbegin() const {
    return missingIndicesPtr_->getView().indices_cbegin();
}

MissingFeatureVector::missing_index_const_iterator MissingFeatureVector::missing_indices_cend() const {
    return missingIndicesPtr_->getView().indices_cend();
}

void MissingFeatureVector::addMissingIndex(uint32 index) {
    missingIndicesPtr_->getView().set(index, true);
}

bool MissingFeatureVector::isMissing(uint32 index) const {
    return (missingIndicesPtr_->getView())[index];
}

void MissingFeatureVector::clearMissingIndices() {
    missingIndicesPtr_->clear();
}
