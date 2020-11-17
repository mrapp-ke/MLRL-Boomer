#include "feature_vector.h"


FeatureVector::FeatureVector(uint32 numElements)
    : SparseArrayVector<float32>(numElements) {

}

FeatureVector::missing_index_const_iterator FeatureVector::missing_indices_cbegin() const {
    return missingIndices_.indices_cbegin();
}

FeatureVector::missing_index_const_iterator FeatureVector::missing_indices_cend() const {
    return missingIndices_.indices_cend();
}

void FeatureVector::addMissingIndex(uint32 index) {
    missingIndices_.setValue(index);
}

void FeatureVector::clearMissingIndices() {
    missingIndices_.setAllToZero();
}
