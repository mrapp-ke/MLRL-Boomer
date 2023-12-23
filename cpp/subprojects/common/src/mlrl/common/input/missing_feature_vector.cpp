#include "mlrl/common/input/missing_feature_vector.hpp"

OldMissingFeatureVector::OldMissingFeatureVector()
    : missingIndicesPtr_(
      std::make_unique<ClearableViewDecorator<ViewDecorator<AllocatedBinaryDokVector>>>(AllocatedBinaryDokVector())) {}

OldMissingFeatureVector::OldMissingFeatureVector(OldMissingFeatureVector& missingFeatureVector)
    : missingIndicesPtr_(std::move(missingFeatureVector.missingIndicesPtr_)) {}

OldMissingFeatureVector::missing_index_const_iterator OldMissingFeatureVector::missing_indices_cbegin() const {
    return missingIndicesPtr_->getView().indices_cbegin();
}

OldMissingFeatureVector::missing_index_const_iterator OldMissingFeatureVector::missing_indices_cend() const {
    return missingIndicesPtr_->getView().indices_cend();
}

void OldMissingFeatureVector::addMissingIndex(uint32 index) {
    missingIndicesPtr_->getView().set(index, true);
}

bool OldMissingFeatureVector::isMissing(uint32 index) const {
    return (missingIndicesPtr_->getView())[index];
}

void OldMissingFeatureVector::clearMissingIndices() {
    missingIndicesPtr_->clear();
}
