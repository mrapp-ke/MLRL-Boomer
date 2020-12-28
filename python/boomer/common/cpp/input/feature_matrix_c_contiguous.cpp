#include "feature_matrix_c_contiguous.h"


CContiguousFeatureMatrix::CContiguousFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* x)
    : numExamples_(numExamples), numFeatures_(numFeatures), x_(x) {

}

CContiguousFeatureMatrix::const_iterator CContiguousFeatureMatrix::row_cbegin(uint32 row) const {
    return &x_[row * numFeatures_];
}

CContiguousFeatureMatrix::const_iterator CContiguousFeatureMatrix::row_cend(uint32 row) const {
    return &x_[(row + 1) * numFeatures_];
}

uint32 CContiguousFeatureMatrix::getNumExamples() const {
    return numExamples_;
}

uint32 CContiguousFeatureMatrix::getNumFeatures() const {
    return numFeatures_;
}
