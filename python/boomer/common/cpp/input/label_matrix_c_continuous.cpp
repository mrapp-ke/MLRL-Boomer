#include "label_matrix_c_continuous.h"


CContinuousLabelMatrix::CContinuousLabelMatrix(uint32 numExamples, uint32 numLabels, const uint8* y)
    : numExamples_(numExamples), numLabels_(numLabels), y_(y) {

}

uint32 CContinuousLabelMatrix::getNumExamples() const {
    return numExamples_;
}

uint32 CContinuousLabelMatrix::getNumLabels() const {
    return numLabels_;
}

uint8 CContinuousLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    uint32 i = (exampleIndex * numLabels_) + labelIndex;
    return y_[i];
}
