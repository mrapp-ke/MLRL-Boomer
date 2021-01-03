#include "label_matrix_c_contiguous.h"


CContiguousLabelMatrix::CContiguousLabelMatrix(uint32 numExamples, uint32 numLabels, const uint8* y)
    : numExamples_(numExamples), numLabels_(numLabels), y_(y) {

}

uint32 CContiguousLabelMatrix::getNumExamples() const {
    return numExamples_;
}

uint32 CContiguousLabelMatrix::getNumLabels() const {
    return numLabels_;
}

uint8 CContiguousLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    uint32 i = (exampleIndex * numLabels_) + labelIndex;
    return y_[i];
}
