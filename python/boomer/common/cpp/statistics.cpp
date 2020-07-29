#include "statistics.h"

using namespace statistics;


AbstractLabelMatrix::AbstractLabelMatrix(intp numExamples, intp numLabels) {
    numExamples_ = numExamples;
    numLabels_ = numLabels;
}

AbstractLabelMatrix::~AbstractLabelMatrix() {

}

uint8 AbstractLabelMatrix::getLabel(intp exampleIndex, intp labelIndex) {
    return 0;
}

DenseLabelMatrixImpl::DenseLabelMatrixImpl(intp numExamples, intp numLabels, uint8* y)
    : AbstractLabelMatrix(numExamples, numLabels) {
    y_ = y;
}

DenseLabelMatrixImpl::~DenseLabelMatrixImpl() {

}

uint8 DenseLabelMatrixImpl::getLabel(intp exampleIndex, intp labelIndex) {
    intp i = (exampleIndex * numExamples_) + labelIndex;
    return y_[i];
}

DokLabelMatrixImpl::DokLabelMatrixImpl(intp numExamples, intp numLabels)
    : AbstractLabelMatrix(numExamples, numLabels) {

}

DokLabelMatrixImpl::~DokLabelMatrixImpl() {

}

uint8 DokLabelMatrixImpl::getLabel(intp exampleIndex, intp labelIndex) {
    return 0;
}
