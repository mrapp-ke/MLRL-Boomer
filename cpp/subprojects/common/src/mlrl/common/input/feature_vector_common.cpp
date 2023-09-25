#include "mlrl/common/input/feature_vector_common.hpp"

AbstractFeatureVector::missing_index_const_iterator AbstractFeatureVector::missing_indices_cbegin() const {
    return missingIndices_.indices_cbegin();
}

AbstractFeatureVector::missing_index_const_iterator AbstractFeatureVector::missing_indices_cend() const {
    return missingIndices_.indices_cend();
}

void AbstractFeatureVector::setMissing(uint32 index, bool missing) {
    missingIndices_.set(index, missing);
}

bool AbstractFeatureVector::isMissing(uint32 index) const {
    return missingIndices_[index];
}
