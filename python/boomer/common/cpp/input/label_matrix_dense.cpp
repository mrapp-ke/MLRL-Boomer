#include "label_matrix_dense.h"


DenseLabelMatrix::DenseLabelMatrix(uint32 numExamples, uint32 numLabels, const uint8* y)
    : numExamples_(numExamples), numLabels_(numLabels), y_(y) {

}

uint32 DenseLabelMatrix::getNumExamples() const {
    return numExamples_;
}

uint32 DenseLabelMatrix::getNumLabels() const {
    return numLabels_;
}

uint8 DenseLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    uint32 i = (exampleIndex * numLabels_) + labelIndex;
    return y_[i];
}
