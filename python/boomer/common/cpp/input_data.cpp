#include "input_data.h"
#include<stdlib.h>


DenseLabelMatrixImpl::DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y) {
    numExamples_ = numExamples;
    numLabels_ = numLabels;
    y_ = y;
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

DokLabelMatrixImpl::DokLabelMatrixImpl(std::unique_ptr<BinaryDokMatrix> matrixPtr) {
    matrixPtr_ = std::move(matrixPtr);
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

DenseFeatureMatrixImpl::DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) {
    numExamples_ = numExamples;
    numFeatures_ = numFeatures;
    x_ = x;
}

uint32 DenseFeatureMatrixImpl::getNumRows() const {
    return numExamples_;
}

uint32 DenseFeatureMatrixImpl::getNumCols() const {
    return numFeatures_;
}

void DenseFeatureMatrixImpl::fetchFeatureValues(uint32 featureIndex, IndexedFloat32Array& indexedArray) const {
    // The number of elements to be returned
    uint32 numElements = this->getNumRows();
    // The array that stores the indices
    IndexedFloat32* array = (IndexedFloat32*) malloc(numElements * sizeof(IndexedFloat32));
    // The first element in `x_` that corresponds to the given feature index
    uint32 offset = featureIndex * numElements;

    for (uint32 i = 0; i < numElements; i++) {
        array[i].index = i;
        array[i].value = x_[offset + i];
    }

    // Update the given struct...
    indexedArray.numElements = numElements;
    indexedArray.data = array;
}

CscFeatureMatrixImpl::CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData,
                                           const uint32* xRowIndices, const uint32* xColIndices)  {
    numExamples_ = numExamples;
    numFeatures_ = numFeatures;
    xData_ = xData;
    xRowIndices_ = xRowIndices;
    xColIndices_ = xColIndices;
}

uint32 CscFeatureMatrixImpl::getNumRows() const {
    return numExamples_;
}

uint32 CscFeatureMatrixImpl::getNumCols() const {
    return numFeatures_;
}

void CscFeatureMatrixImpl::fetchFeatureValues(uint32 featureIndex, IndexedFloat32Array& indexedArray) const {
    // The index of the first element in `xData_` and `xRowIndices_` that corresponds to the given feature index+
    uint32 start = xColIndices_[featureIndex];
    // The index of the last element in `xData_` and `xRowIndices_` that corresponds to the given feature index
    uint32 end = xColIndices_[featureIndex + 1];
    // The number of elements to be returned
    uint32 numElements = end - start;
    // The array that stores the indices
    IndexedFloat32* array = NULL;

    if (numElements > 0) {
        array = (IndexedFloat32*) malloc(numElements * sizeof(IndexedFloat32));
        uint32 i = 0;

        for (uint32 j = start; j < end; j++) {
            array[i].index = xRowIndices_[j];
            array[i].value = xData_[j];
            i++;
        }
    }

    // Update the given struct...
    indexedArray.numElements = numElements;
    indexedArray.data = array;
}

DokNominalFeatureVectorImpl::DokNominalFeatureVectorImpl(std::unique_ptr<BinaryDokVector> vectorPtr) {
    vectorPtr_ = std::move(vectorPtr);
}

uint32 DokNominalFeatureVectorImpl::getNumElements() const {
    return vectorPtr_->getNumElements();
}

bool DokNominalFeatureVectorImpl::hasZeroElements() const {
    return vectorPtr_->hasZeroElements();
}

uint8 DokNominalFeatureVectorImpl::getValue(uint32 pos) const {
    return vectorPtr_->getValue(pos);
}
