#include "feature_matrix_dense.h"


DenseFeatureMatrix::DenseFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* x)
    : numExamples_(numExamples), numFeatures_(numFeatures), x_(x) {

}

uint32 DenseFeatureMatrix::getNumExamples() const {
    return numExamples_;
}

uint32 DenseFeatureMatrix::getNumFeatures() const {
    return numFeatures_;
}

void DenseFeatureMatrix::fetchFeatureVector(uint32 featureIndex,
                                            std::unique_ptr<FeatureVector>& featureVectorPtr) const {
    // The number of elements to be returned
    uint32 numElements = this->getNumExamples();
    // The first element in `x_` that corresponds to the given feature index
    uint32 offset = featureIndex * numElements;

    featureVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator iterator = featureVectorPtr->begin();
    uint32 i = 0;

    for (uint32 j = 0; j < numElements; j++) {
        float32 value = x_[offset + j];

        if (value != value) {
            // The value is NaN (because comparisons to NaN always evaluate to false)...
            featureVectorPtr->addMissingIndex(j);
        } else {
            iterator[i].index = j;
            iterator[i].value = value;
            i++;
        }
    }

    featureVectorPtr->setNumElements(i, true);
}
