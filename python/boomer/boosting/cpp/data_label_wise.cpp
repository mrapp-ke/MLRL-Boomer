#include "data_label_wise.h"
#include <cstdlib>

using namespace boosting;


DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements)
    : DenseLabelWiseStatisticVector(numElements, false) {

}

DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements, bool init)
    : numElements_(numElements),
      gradients_((float64*) (init ? calloc(numElements, sizeof(float64)) : malloc(numElements * sizeof(float64)))),
      hessians_((float64*) (init ? calloc(numElements, sizeof(float64)) : malloc(numElements * sizeof(float64)))) {

}

DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& vector)
    : DenseLabelWiseStatisticVector(vector.numElements_) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] = vector.gradients_[i];
        hessians_[i] = vector.hessians_[i];
    }
}

DenseLabelWiseStatisticVector::~DenseLabelWiseStatisticVector() {
    free(gradients_);
    free(hessians_);
}

DenseLabelWiseStatisticVector::gradient_iterator DenseLabelWiseStatisticVector::gradients_begin() {
    return gradients_;
}

DenseLabelWiseStatisticVector::gradient_iterator DenseLabelWiseStatisticVector::gradients_end() {
    return &gradients_[numElements_];
}

DenseLabelWiseStatisticVector::gradient_const_iterator DenseLabelWiseStatisticVector::gradients_cbegin() const {
    return gradients_;
}

DenseLabelWiseStatisticVector::gradient_const_iterator DenseLabelWiseStatisticVector::gradients_cend() const {
    return &gradients_[numElements_];
}

DenseLabelWiseStatisticVector::hessian_iterator DenseLabelWiseStatisticVector::hessians_begin() {
    return hessians_;
}

DenseLabelWiseStatisticVector::hessian_iterator DenseLabelWiseStatisticVector::hessians_end() {
    return &hessians_[numElements_];
}

DenseLabelWiseStatisticVector::hessian_const_iterator DenseLabelWiseStatisticVector::hessians_cbegin() const {
    return hessians_;
}

DenseLabelWiseStatisticVector::hessian_const_iterator DenseLabelWiseStatisticVector::hessians_cend() const {
    return &hessians_[numElements_];
}

void DenseLabelWiseStatisticVector::setAllToZero() {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] = 0;
        hessians_[i] = 0;
    }
}

void DenseLabelWiseStatisticVector::add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                                        hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] += gradientsBegin[i];
        hessians_[i] += hessiansBegin[i];
    }
}

void DenseLabelWiseStatisticVector::add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                                        hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                                        float64 weight) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] += (gradientsBegin[i] * weight);
        hessians_[i] += (hessiansBegin[i] * weight);
    }
}

void DenseLabelWiseStatisticVector::subtract(gradient_const_iterator gradientsBegin,
                                             gradient_const_iterator gradientsEnd,
                                             hessian_const_iterator hessiansBegin,
                                             hessian_const_iterator hessiansEnd, float64 weight) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] -= (gradientsBegin[i] * weight);
        hessians_[i] -= (hessiansBegin[i] * weight);
    }
}

void DenseLabelWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                gradient_const_iterator gradientsEnd,
                                                hessian_const_iterator hessiansBegin,
                                                hessian_const_iterator hessiansEnd, const FullIndexVector& indices,
                                                float64 weight) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] += (gradientsBegin[i] * weight);
        hessians_[i] += (hessiansBegin[i] * weight);
    }
}

void DenseLabelWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                gradient_const_iterator gradientsEnd,
                                                hessian_const_iterator hessiansBegin,
                                                hessian_const_iterator hessiansEnd, const PartialIndexVector& indices,
                                                float64 weight) {
    PartialIndexVector::const_iterator indexIterator = indices.cbegin();

    for (uint32 i = 0; i < numElements_; i++) {
        uint32 index = indexIterator[i];
        gradients_[i] += (gradientsBegin[index] * weight);
        hessians_[i] += (hessiansBegin[index] * weight);
    }
}

void DenseLabelWiseStatisticVector::difference(gradient_const_iterator firstGradientsBegin,
                                               gradient_const_iterator firstGradientsEnd,
                                               hessian_const_iterator firstHessiansBegin,
                                               hessian_const_iterator firstHessiansEnd,
                                               const FullIndexVector& firstIndices,
                                               gradient_const_iterator secondGradientsBegin,
                                               gradient_const_iterator secondGradientsEnd,
                                               hessian_const_iterator secondHessiansBegin,
                                               hessian_const_iterator secondHessiansEnd) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] = firstGradientsBegin[i] - secondGradientsBegin[i];
        hessians_[i] = firstHessiansBegin[i] - secondHessiansBegin[i];
    }
}

void DenseLabelWiseStatisticVector::difference(gradient_const_iterator firstGradientsBegin,
                                               gradient_const_iterator firstGradientsEnd,
                                               hessian_const_iterator firstHessiansBegin,
                                               hessian_const_iterator firstHessiansEnd,
                                               const PartialIndexVector& firstIndices,
                                               gradient_const_iterator secondGradientsBegin,
                                               gradient_const_iterator secondGradientsEnd,
                                               hessian_const_iterator secondHessiansBegin,
                                               hessian_const_iterator secondHessiansEnd) {
    PartialIndexVector::const_iterator firstIndexIterator = firstIndices.cbegin();

    for (uint32 i = 0; i < numElements_; i++) {
        uint32 firstIndex = firstIndexIterator[i];
        gradients_[i] = firstGradientsBegin[firstIndex] - secondGradientsBegin[i];
        hessians_[i] = firstHessiansBegin[firstIndex] - secondHessiansBegin[i];
    }
}

DenseLabelWiseStatisticMatrix::DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
    : DenseLabelWiseStatisticMatrix(numRows, numCols, false) {

}

DenseLabelWiseStatisticMatrix::DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols, bool init)
    : numRows_(numRows), numCols_(numCols),
      gradients_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                  : malloc(numRows * numCols * sizeof(float64)))),
      hessians_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                 : malloc(numRows * numCols * sizeof(float64)))) {

}
            
DenseLabelWiseStatisticMatrix::~DenseLabelWiseStatisticMatrix() {
    free(gradients_);
    free(hessians_);
}

DenseLabelWiseStatisticMatrix::gradient_iterator DenseLabelWiseStatisticMatrix::gradients_row_begin(uint32 row) {
    return &gradients_[row * numCols_];
}
            
DenseLabelWiseStatisticMatrix::gradient_iterator DenseLabelWiseStatisticMatrix::gradients_row_end(uint32 row) {
    return &gradients_[(row + 1) * numCols_];
}
            
DenseLabelWiseStatisticMatrix::gradient_const_iterator DenseLabelWiseStatisticMatrix::gradients_row_cbegin(
        uint32 row) const {
    return &gradients_[row * numCols_];
}
            
DenseLabelWiseStatisticMatrix::gradient_const_iterator DenseLabelWiseStatisticMatrix::gradients_row_cend(
        uint32 row) const {
    return &gradients_[(row + 1) * numCols_];
}

DenseLabelWiseStatisticVector::hessian_iterator DenseLabelWiseStatisticMatrix::hessians_row_begin(uint32 row) {
    return &hessians_[row * numCols_];
}
            
DenseLabelWiseStatisticVector::hessian_iterator DenseLabelWiseStatisticMatrix::hessians_row_end(uint32 row) {
    return &hessians_[(row + 1) * numCols_];
}
            
DenseLabelWiseStatisticMatrix::hessian_const_iterator DenseLabelWiseStatisticMatrix::hessians_row_cbegin(
        uint32 row) const {
    return &hessians_[row * numCols_];
}
            
DenseLabelWiseStatisticMatrix::hessian_const_iterator DenseLabelWiseStatisticMatrix::hessians_row_cend(
        uint32 row) const {
    return &hessians_[(row + 1) * numCols_];
}

void DenseLabelWiseStatisticMatrix::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                             gradient_const_iterator gradientsEnd, hessian_const_iterator hessiansBegin,
                                             hessian_const_iterator hessiansEnd) {
    uint32 offset = row * numCols_;

    for (uint32 i = 0; i < numCols_; i++) {
        uint32 index = offset + i;
        gradients_[index] += gradientsBegin[i];
        hessians_[index] += hessiansBegin[i];
    }
}
