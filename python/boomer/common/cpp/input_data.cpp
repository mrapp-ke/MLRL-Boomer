#include "input_data.h"


AbstractLabelMatrix::AbstractLabelMatrix(intp numExamples, intp numLabels) {
    numExamples_ = numExamples;
    numLabels_ = numLabels;
}

AbstractLabelMatrix::~AbstractLabelMatrix() {

}

uint8 AbstractLabelMatrix::getLabel(intp exampleIndex, intp labelIndex) {
    return 0;
}

DenseLabelMatrixImpl::DenseLabelMatrixImpl(intp numExamples, intp numLabels, const uint8* y)
    : AbstractLabelMatrix(numExamples, numLabels) {
    y_ = y;
}

DenseLabelMatrixImpl::~DenseLabelMatrixImpl() {

}

uint8 DenseLabelMatrixImpl::getLabel(intp exampleIndex, intp labelIndex) {
    intp i = (exampleIndex * numLabels_) + labelIndex;
    return y_[i];
}

DokLabelMatrixImpl::DokLabelMatrixImpl(intp numExamples, intp numLabels, std::shared_ptr<BinaryDokMatrix> dokMatrixPtr)
    : AbstractLabelMatrix(numExamples, numLabels) {
    dokMatrixPtr_ = dokMatrixPtr;
}

DokLabelMatrixImpl::~DokLabelMatrixImpl() {

}

uint8 DokLabelMatrixImpl::getLabel(intp exampleIndex, intp labelIndex) {
    return dokMatrixPtr_.get()->getValue((uint32) exampleIndex, (uint32) labelIndex);
}
