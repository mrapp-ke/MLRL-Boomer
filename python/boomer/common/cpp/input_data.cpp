#include "input_data.h"
#include<stdlib.h>


DenseLabelMatrixImpl::DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y) {
    numExamples_ = numExamples;
    numLabels_ = numLabels;
    y_ = y;
}

uint32 DenseLabelMatrixImpl::getNumRows() {
    return numExamples_;
}

uint32 DenseLabelMatrixImpl::getNumCols() {
    return numLabels_;
}

uint8 DenseLabelMatrixImpl::getValue(uint32 row, uint32 col) {
    uint32 i = (row * this->getNumCols()) + col;
    return y_[i];
}

DokLabelMatrixImpl::DokLabelMatrixImpl(BinaryDokMatrix* matrix) {
    matrix_ = matrix;
}

DokLabelMatrixImpl::~DokLabelMatrixImpl() {
    delete matrix_;
}

uint32 DokLabelMatrixImpl::getNumRows() {
    return matrix_->getNumRows();
}

uint32 DokLabelMatrixImpl::getNumCols() {
    return matrix_->getNumCols();
}

uint8 DokLabelMatrixImpl::getValue(uint32 row, uint32 col) {
    return matrix_->getValue(row, col);
}

DenseFeatureMatrixImpl::DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x) {
    numExamples_ = numExamples;
    numFeatures_ = numFeatures;
    x_ = x;
}

uint32 DenseFeatureMatrixImpl::getNumRows() {
    return numExamples_;
}

uint32 DenseFeatureMatrixImpl::getNumCols() {
    return numFeatures_;
}

void DenseFeatureMatrixImpl::fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) {
    // The number of elements to be returned
    uint32 numElements = this->getNumRows();
    // The array that stores the indices
    IndexedFloat32* sortedArray = (IndexedFloat32*) malloc(numElements * sizeof(IndexedFloat32));
    // The first element in `x_` that corresponds to the given feature index
    uint32 offset = featureIndex * numElements;

    for (uint32 i = 0; i < numElements; i++) {
        sortedArray[i].index = i;
        sortedArray[i].value = x_[offset + i];
    }

    // Sort the array...
    qsort(sortedArray, numElements, sizeof(IndexedFloat32), &tuples::compareIndexedFloat32);

    // Update the given struct...
    indexedArray->numElements = numElements;
    indexedArray->data = sortedArray;
}

void DenseFeatureMatrixImpl::fetchFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) {
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
    indexedArray->numElements = numElements;
    indexedArray->data = array;
}

CscFeatureMatrixImpl::CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData,
                                           const uint32* xRowIndices, const uint32* xColIndices)  {
    numExamples_ = numExamples;
    numFeatures_ = numFeatures;
    xData_ = xData;
    xRowIndices_ = xRowIndices;
    xColIndices_ = xColIndices;
}

uint32 CscFeatureMatrixImpl::getNumRows() {
    return numExamples_;
}

uint32 CscFeatureMatrixImpl::getNumCols() {
    return numFeatures_;
}

void CscFeatureMatrixImpl::fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) {
    // The index of the first element in `xData_` and `xRowIndices_` that corresponds to the given feature index+
    uint32 start = xColIndices_[featureIndex];
    // The index of the last element in `xData_` and `xRowIndices_` that corresponds to the given feature index
    uint32 end = xColIndices_[featureIndex + 1];
    // The number of elements to be returned
    uint32 numElements = end - start;
    // The array that stores the indices
    IndexedFloat32* sortedArray = NULL;

    if (numElements > 0) {
        sortedArray = (IndexedFloat32*) malloc(numElements * sizeof(IndexedFloat32));
        uint32 i = 0;

        for (uint32 j = start; j < end; j++) {
            sortedArray[i].index = xRowIndices_[j];
            sortedArray[i].value = xData_[j];
            i++;
        }

        // Sort the array...
        qsort(sortedArray, numElements, sizeof(IndexedFloat32), &tuples::compareIndexedFloat32);
    }

    // Update the given struct...
    indexedArray->numElements = numElements;
    indexedArray->data = sortedArray;
}

void CscFeatureMatrixImpl::fetchFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) {
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
    indexedArray->numElements = numElements;
    indexedArray->data = array;
}

DokNominalFeatureVectorImpl::DokNominalFeatureVectorImpl(BinaryDokVector* vector) {
    vector_ = vector;
}

DokNominalFeatureVectorImpl::~DokNominalFeatureVectorImpl() {
    delete vector_;
}

uint32 DokNominalFeatureVectorImpl::getNumElements() {
    return vector_->getNumElements();
}

bool DokNominalFeatureVectorImpl::hasZeroElements() {
    return vector_->hasZeroElements();
}

uint8 DokNominalFeatureVectorImpl::getValue(uint32 pos) {
    return vector_->getValue(pos);
}
