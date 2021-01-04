#include "label_matrix_dok.h"


DokLabelMatrix::DokLabelMatrix(uint32 numExamples, uint32 numLabels)
    : numExamples_(numExamples), numLabels_(numLabels) {

}

uint32 DokLabelMatrix::getNumRows() const {
    return numExamples_;
}

uint32 DokLabelMatrix::getNumCols() const {
    return numLabels_;
}

uint8 DokLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    return matrix_.getValue(exampleIndex, labelIndex);
}

void DokLabelMatrix::setValue(uint32 exampleIndex, uint32 labelIndex) {
    matrix_.setValue(exampleIndex, labelIndex);
}
