#include "common/input/label_matrix_dok.hpp"


DokLabelMatrix::DokLabelMatrix(uint32 numRows, uint32 numCols)
    : numRows_(numRows), numCols_(numCols) {

}

uint32 DokLabelMatrix::getNumRows() const {
    return numRows_;
}

uint32 DokLabelMatrix::getNumCols() const {
    return numCols_;
}

std::unique_ptr<LabelVector> DokLabelMatrix::getLabelVector(uint32 row) const {
    std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numCols_);
    LabelVector::index_iterator iterator = labelVectorPtr->indices_begin();
    uint32 n = 0;

    for (uint32 i = 0; i < numCols_; i++) {
        if (this->getValue(row, i)) {
            iterator[n] = i;
            n++;
        }
    }

    labelVectorPtr->setNumElements(n, true);
    return labelVectorPtr;
}

uint8 DokLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    return matrix_.getValue(exampleIndex, labelIndex);
}

void DokLabelMatrix::setValue(uint32 exampleIndex, uint32 labelIndex) {
    matrix_.setValue(exampleIndex, labelIndex);
}
