#include "feature_matrix_csc.h"


CscFeatureMatrix::CscFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* xData,
                                   const uint32* xRowIndices, const uint32* xColIndices)
    : numExamples_(numExamples), numFeatures_(numFeatures), xData_(xData), xRowIndices_(xRowIndices),
      xColIndices_(xColIndices) {

}

uint32 CscFeatureMatrix::getNumExamples() const {
    return numExamples_;
}

uint32 CscFeatureMatrix::getNumFeatures() const {
    return numFeatures_;
}

void CscFeatureMatrix::fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const {
    // The index of the first element in `xData_` and `xRowIndices_` that corresponds to the given feature index
    uint32 start = xColIndices_[featureIndex];
    // The index of the last element in `xData_` and `xRowIndices_` that corresponds to the given feature index
    uint32 end = xColIndices_[featureIndex + 1];
    // The number of elements to be returned
    uint32 numElements = end - start;

    featureVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator iterator = featureVectorPtr->begin();
    uint32 i = 0;

    for (uint32 j = start; j < end; j++) {
        uint32 index = xRowIndices_[j];
        float32 value = xData_[j];

        if (value != value) {
            // The value is NaN (because comparisons to NaN always evaluate to false)...
            featureVectorPtr->addMissingIndex(index);
        } else {
            iterator[i].index = index;
            iterator[i].value = value;
            i++;
        }
    }

    featureVectorPtr->setNumElements(i, true);
}
