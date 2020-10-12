#include "input_data.h"
#include<stdlib.h>


DenseLabelMatrixImpl::DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y)
    : numExamples_(numExamples), numLabels_(numLabels), y_(y) {

}

uint32 DenseLabelMatrixImpl::getNumRows() const {
    return numExamples_;
}

uint32 DenseLabelMatrixImpl::getNumCols() const {
    return numLabels_;
}

uint8 DenseLabelMatrixImpl::getValue(uint32 row, uint32 col) const {
    uint32 i = (row * this->getNumCols()) + col;
    return y_[i];
}

DokLabelMatrixImpl::DokLabelMatrixImpl(std::unique_ptr<BinaryDokMatrix> matrixPtr)
    : matrixPtr_(std::move(matrixPtr)) {

}

uint32 DokLabelMatrixImpl::getNumRows() const {
    return matrixPtr_->getNumRows();
}

uint32 DokLabelMatrixImpl::getNumCols() const {
    return matrixPtr_->getNumCols();
}

uint8 DokLabelMatrixImpl::getValue(uint32 row, uint32 col) const {
    return matrixPtr_->getValue(row, col);
}

DenseFeatureMatrixImpl::DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x)
    : numExamples_(numExamples), numFeatures_(numFeatures), x_(x) {

}

uint32 DenseFeatureMatrixImpl::getNumRows() const {
    return numExamples_;
}

uint32 DenseFeatureMatrixImpl::getNumCols() const {
    return numFeatures_;
}

void DenseFeatureMatrixImpl::fetchFeatureVector(uint32 featureIndex,
                                                std::unique_ptr<FeatureVector>& featureVectorPtr) const {
    // The number of elements to be returned
    uint32 numElements = this->getNumRows();
    // The first element in `x_` that corresponds to the given feature index
    uint32 offset = featureIndex * numElements;

    featureVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator iterator = featureVectorPtr->begin();

    for (uint32 i = 0; i < numElements; i++) {
        iterator[i].index = i;
        iterator[i].value = x_[offset + i];
    }
}

CscFeatureMatrixImpl::CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData,
                                           const uint32* xRowIndices, const uint32* xColIndices)
    : numExamples_(numExamples), numFeatures_(numFeatures), xData_(xData), xRowIndices_(xRowIndices),
      xColIndices_(xColIndices) {

}

uint32 CscFeatureMatrixImpl::getNumRows() const {
    return numExamples_;
}

uint32 CscFeatureMatrixImpl::getNumCols() const {
    return numFeatures_;
}

void CscFeatureMatrixImpl::fetchFeatureVector(uint32 featureIndex,
                                              std::unique_ptr<FeatureVector>& featureVectorPtr) const {
    // The index of the first element in `xData_` and `xRowIndices_` that corresponds to the given feature index
    uint32 start = xColIndices_[featureIndex];
    // The index of the last element in `xData_` and `xRowIndices_` that corresponds to the given feature index
    uint32 end = xColIndices_[featureIndex + 1];
    // The number of elements to be returned
    uint32 numElements = end - start;

    featureVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator iterator = featureVectorPtr->begin();

    for (uint32 j = start; j < end; j++) {
        iterator->index = xRowIndices_[j];
        iterator->value = xData_[j];
        iterator++;
    }
}

DokNominalFeatureVectorImpl::DokNominalFeatureVectorImpl(std::unique_ptr<BinaryDokVector> vectorPtr)
    : vectorPtr_(std::move(vectorPtr)) {

}

uint32 DokNominalFeatureVectorImpl::getNumElements() const {
    return vectorPtr_->getNumElements();
}

uint8 DokNominalFeatureVectorImpl::getValue(uint32 pos) const {
    return vectorPtr_->getValue(pos);
}
