#include "input_data.h"


AbstractLabelMatrix::AbstractLabelMatrix(uint32 numExamples, uint32 numLabels) {
    numExamples_ = numExamples;
    numLabels_ = numLabels;
}

uint32 AbstractLabelMatrix::getNumRows() {
    return numExamples_;
}

uint32 AbstractLabelMatrix::getNumCols() {
    return numLabels_;
}

AbstractRandomAccessLabelMatrix::AbstractRandomAccessLabelMatrix(uint32 numExamples, uint32 numLabels)
    : AbstractLabelMatrix(numExamples, numLabels) {

}

uint8 AbstractRandomAccessLabelMatrix::getLabel(uint32 exampleIndex, uint32 labelIndex) {
    return 0;
}

DenseLabelMatrixImpl::DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y)
    : AbstractRandomAccessLabelMatrix(numExamples, numLabels) {
    y_ = y;
}

DenseLabelMatrixImpl::~DenseLabelMatrixImpl() {

}

uint8 DenseLabelMatrixImpl::getLabel(uint32 exampleIndex, uint32 labelIndex) {
    uint32 i = (exampleIndex * this->getNumCols()) + labelIndex;
    return y_[i];
}

DokLabelMatrixImpl::DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels,
                                       std::shared_ptr<BinaryDokMatrix> dokMatrixPtr)
    : AbstractRandomAccessLabelMatrix(numExamples, numLabels) {
    dokMatrixPtr_ = dokMatrixPtr;
}

DokLabelMatrixImpl::~DokLabelMatrixImpl() {

}

uint8 DokLabelMatrixImpl::getLabel(uint32 exampleIndex, uint32 labelIndex) {
    return dokMatrixPtr_.get()->getValue(exampleIndex, labelIndex);
}

AbstractFeatureMatrix::AbstractFeatureMatrix(uint32 numExamples, uint32 numFeatures) {
    numExamples_ = numExamples;
    numFeatures_ = numFeatures;
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
    // The total number of features
    uint32 numFeatures = this->getNumCols();
    // The array that stores the indices
    IndexedFloat32* sortedArray = (IndexedFloat32*) malloc(numElements * sizeof(IndexedFloat32));

    for (uint32 r = 0; r < numElements; r++) {
        uint32 i = (r * numFeatures) + featureIndex;
        sortedArray[r].index = r;
        sortedArray[r].value = x_[i];
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
