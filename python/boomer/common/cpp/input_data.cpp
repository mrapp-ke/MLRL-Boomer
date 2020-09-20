#include "input_data.h"


AbstractLabelMatrix::~AbstractLabelMatrix() {

}

DenseLabelMatrixImpl::DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y) {
    numExamples_ = numExamples;
    numLabels_ = numLabels;
    y_ = y;
}

DenseLabelMatrixImpl::~DenseLabelMatrixImpl() {

}

uint32 DenseLabelMatrixImpl::getNumRows() {
    return numExamples_;
}

uint32 DenseLabelMatrixImpl::getNumCols() {
    return numLabels_;
}

uint8 DenseLabelMatrixImpl::get(uint32 row, uint32 col) {
    uint32 i = (row * this->getNumCols()) + col;
    return y_[i];
}

DokLabelMatrixImpl::DokLabelMatrixImpl(std::shared_ptr<BinaryDokMatrix> dokMatrixPtr) {
    dokMatrixPtr_ = dokMatrixPtr;
}

DokLabelMatrixImpl::~DokLabelMatrixImpl() {

}

uint32 DokLabelMatrixImpl::getNumRows() {
    return dokMatrixPtr_.get()->getNumRows();
}

uint32 DokLabelMatrixImpl::getNumCols() {
    return dokMatrixPtr_.get()->getNumCols();
}

uint8 DokLabelMatrixImpl::get(uint32 row, uint32 col) {
    return dokMatrixPtr_.get()->get(row, col);
}

AbstractFeatureMatrix::AbstractFeatureMatrix(uint32 numExamples, uint32 numFeatures) {
    numExamples_ = numExamples;
    numFeatures_ = numFeatures;
}

AbstractFeatureMatrix::~AbstractFeatureMatrix() {

}

uint32 AbstractFeatureMatrix::getNumRows() {
    return numExamples_;
}

uint32 AbstractFeatureMatrix::getNumCols() {
    return numFeatures_;
}

void AbstractFeatureMatrix::fetchSortedFeatureValues(uint32 featureIndex, IndexedFloat32Array* indexedArray) {

}

DenseFeatureMatrixImpl::DenseFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* x)
    : AbstractFeatureMatrix(numExamples, numFeatures) {
    x_ = x;
}

DenseFeatureMatrixImpl::~DenseFeatureMatrixImpl() {

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

CscFeatureMatrixImpl::CscFeatureMatrixImpl(uint32 numExamples, uint32 numFeatures, const float32* xData,
                                           const uint32* xRowIndices, const uint32* xColIndices)
    : AbstractFeatureMatrix(numExamples, numFeatures) {
    xData_ = xData;
    xRowIndices_ = xRowIndices;
    xColIndices_ = xColIndices;
}

CscFeatureMatrixImpl::~CscFeatureMatrixImpl() {

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
