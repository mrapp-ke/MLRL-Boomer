#include "feature_matrix_csr.h"


CsrFeatureMatrix::CsrFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* xData,
                                   const uint32* xRowIndices, const uint32* xColIndices)
    : numExamples_(numExamples), numFeatures_(numFeatures), xData_(xData), xRowIndices_(xRowIndices),
      xColIndices_(xColIndices){

}

CsrFeatureMatrix::value_const_iterator CsrFeatureMatrix::row_values_cbegin(uint32 row) const {
    return &xData_[xRowIndices_[row]];
}

CsrFeatureMatrix::value_const_iterator CsrFeatureMatrix::row_values_cend(uint32 row) const {
    return &xData_[xRowIndices_[row + 1]];
}

CsrFeatureMatrix::index_const_iterator CsrFeatureMatrix::row_indices_cbegin(uint32 row) const {
    return &xColIndices_[xRowIndices_[row]];
}

CsrFeatureMatrix::index_const_iterator CsrFeatureMatrix::row_indices_cend(uint32 row) const {
    return &xColIndices_[xRowIndices_[row + 1]];
}

uint32 CsrFeatureMatrix::getNumExamples() const {
    return numExamples_;
}

uint32 CsrFeatureMatrix::getNumFeatures() const {
    return numFeatures_;
}
