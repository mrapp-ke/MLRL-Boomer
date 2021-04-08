#include "common/input/label_matrix_c_contiguous.hpp"


CContiguousLabelMatrix::CContiguousLabelMatrix(uint32 numRows, uint32 numCols, uint8* array)
    : view_(CContiguousView<uint8>(numRows, numCols, array)) {

}

uint32 CContiguousLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CContiguousLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}

std::unique_ptr<LabelVector> CContiguousLabelMatrix::getLabelVector(uint32 row) const {
    uint32 numCols = this->getNumCols();
    std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numCols);
    LabelVector::index_iterator iterator = labelVectorPtr->indices_begin();
    uint32 n = 0;

    for (uint32 i = 0; i < numCols; i++) {
        if (this->getValue(row, i)) {
            iterator[n] = i;
            n++;
        }
    }

    labelVectorPtr->setNumElements(n, true);
    return labelVectorPtr;
}

uint8 CContiguousLabelMatrix::getValue(uint32 exampleIndex, uint32 labelIndex) const {
    return view_.row_cbegin(exampleIndex)[labelIndex];
}
