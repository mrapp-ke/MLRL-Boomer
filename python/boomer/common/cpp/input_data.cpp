#include "input_data.h"

using namespace input;


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

DokLabelMatrixImpl::DokLabelMatrixImpl(intp numExamples, intp numLabels, sparse::BinaryDokMatrix* dokMatrix)
    : AbstractLabelMatrix(numExamples, numLabels) {
    dokMatrix_ = dokMatrix;
}

DokLabelMatrixImpl::~DokLabelMatrixImpl() {
    delete dokMatrix_;
}

uint8 DokLabelMatrixImpl::getLabel(intp exampleIndex, intp labelIndex) {
    return dokMatrix_->getValue((uint32) exampleIndex, (uint32) labelIndex);
}
