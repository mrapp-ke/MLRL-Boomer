#include "data_label_wise.h"
#include <cstdlib>

using namespace boosting;


DenseLabelWiseStatisticsVector::DenseLabelWiseStatisticsVector(uint32 numElements)
    : DenseLabelWiseStatisticsVector(numElements, false) {

}

DenseLabelWiseStatisticsVector::DenseLabelWiseStatisticsVector(uint32 numElements, bool init)
    : numElements_(numElements),
      gradients_((float64*) (init ? calloc(numElements, sizeof(float64)) : malloc(numElements * sizeof(float64)))),
      hessians_((float64*) (init ? calloc(numElements, sizeof(float64)) : malloc(numElements * sizeof(float64)))) {

}

DenseLabelWiseStatisticsVector::DenseLabelWiseStatisticsVector(const DenseLabelWiseStatisticsVector& vector)
    : DenseLabelWiseStatisticsVector(vector.numElements_) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] = vector.gradients_[i];
        hessians_[i] = vector.hessians_[i];
    }
}

DenseLabelWiseStatisticsVector::~DenseLabelWiseStatisticsVector() {
    free(gradients_);
    free(hessians_);
}

DenseLabelWiseStatisticsVector::gradient_iterator DenseLabelWiseStatisticsVector::gradients_begin() {
    return gradients_;
}

DenseLabelWiseStatisticsVector::gradient_iterator DenseLabelWiseStatisticsVector::gradients_end() {
    return &gradients_[numElements_];
}

DenseLabelWiseStatisticsVector::gradient_const_iterator DenseLabelWiseStatisticsVector::gradients_cbegin() const {
    return gradients_;
}

DenseLabelWiseStatisticsVector::gradient_const_iterator DenseLabelWiseStatisticsVector::gradients_cend() const {
    return &gradients_[numElements_];
}

DenseLabelWiseStatisticsVector::hessian_iterator DenseLabelWiseStatisticsVector::hessians_begin() {
    return hessians_;
}

DenseLabelWiseStatisticsVector::hessian_iterator DenseLabelWiseStatisticsVector::hessians_end() {
    return &hessians_[numElements_];
}

DenseLabelWiseStatisticsVector::hessian_const_iterator DenseLabelWiseStatisticsVector::hessians_cbegin() const {
    return hessians_;
}

DenseLabelWiseStatisticsVector::hessian_const_iterator DenseLabelWiseStatisticsVector::hessians_cend() const {
    return &hessians_[numElements_];
}

uint32 DenseLabelWiseStatisticsVector::getNumElements() const {
    return numElements_;
}

void DenseLabelWiseStatisticsVector::setAllToZero() {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] = 0;
        hessians_[i] = 0;
    }
}

void DenseLabelWiseStatisticsVector::add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                                         hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] += gradientsBegin[i];
        hessians_[i] += hessiansBegin[i];
    }
}

void DenseLabelWiseStatisticsVector::add(gradient_const_iterator gradientsBegin, gradient_const_iterator gradientsEnd,
                                         hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                                         float64 weight) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] += (gradientsBegin[i] * weight);
        hessians_[i] += (hessiansBegin[i] * weight);
    }
}

void DenseLabelWiseStatisticsVector::subtract(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd, float64 weight) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] -= (gradientsBegin[i] * weight);
        hessians_[i] -= (hessiansBegin[i] * weight);
    }
}

void DenseLabelWiseStatisticsVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd, const FullIndexVector& indices,
                                                 float64 weight) {
    for (uint32 i = 0; i < numElements_; i++) {
        gradients_[i] += (gradientsBegin[i] * weight);
        hessians_[i] += (hessiansBegin[i] * weight);
    }
}

void DenseLabelWiseStatisticsVector::addToSubset(gradient_const_iterator gradientsBegin,
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

void DenseLabelWiseStatisticsVector::difference(gradient_const_iterator firstGradientsBegin,
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

void DenseLabelWiseStatisticsVector::difference(gradient_const_iterator firstGradientsBegin,
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

DenseLabelWiseStatisticsMatrix::DenseLabelWiseStatisticsMatrix(uint32 numRows, uint32 numCols)
    : DenseLabelWiseStatisticsMatrix(numRows, numCols, false) {

}

DenseLabelWiseStatisticsMatrix::DenseLabelWiseStatisticsMatrix(uint32 numRows, uint32 numCols, bool init)
    : numRows_(numRows), numCols_(numCols),
      gradients_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                  : malloc(numRows * numCols * sizeof(float64)))),
      hessians_((float64*) (init ? calloc(numRows * numCols, sizeof(float64))
                                 : malloc(numRows * numCols * sizeof(float64)))) {

}
            
DenseLabelWiseStatisticsMatrix::~DenseLabelWiseStatisticsMatrix() {
    free(gradients_);
    free(hessians_);
}

DenseLabelWiseStatisticsMatrix::gradient_iterator DenseLabelWiseStatisticsMatrix::gradients_row_begin(uint32 row) {
    return &gradients_[row * numCols_];
}
            
DenseLabelWiseStatisticsMatrix::gradient_iterator DenseLabelWiseStatisticsMatrix::gradients_row_end(uint32 row) {
    return &gradients_[(row + 1) * numCols_];
}
            
DenseLabelWiseStatisticsMatrix::gradient_const_iterator DenseLabelWiseStatisticsMatrix::gradients_row_cbegin(
        uint32 row) const {
    return &gradients_[row * numCols_];
}
            
DenseLabelWiseStatisticsMatrix::gradient_const_iterator DenseLabelWiseStatisticsMatrix::gradients_row_cend(
        uint32 row) const {
    return &gradients_[(row + 1) * numCols_];
}

DenseLabelWiseStatisticsVector::hessian_iterator DenseLabelWiseStatisticsMatrix::hessians_row_begin(uint32 row) {
    return &hessians_[row * numCols_];
}
            
DenseLabelWiseStatisticsVector::hessian_iterator DenseLabelWiseStatisticsMatrix::hessians_row_end(uint32 row) {
    return &hessians_[(row + 1) * numCols_];
}
            
DenseLabelWiseStatisticsMatrix::hessian_const_iterator DenseLabelWiseStatisticsMatrix::hessians_row_cbegin(
        uint32 row) const {
    return &hessians_[row * numCols_];
}
            
DenseLabelWiseStatisticsMatrix::hessian_const_iterator DenseLabelWiseStatisticsMatrix::hessians_row_cend(
        uint32 row) const {
    return &hessians_[(row + 1) * numCols_];
}

uint32 DenseLabelWiseStatisticsMatrix::getNumRows() const {
    return numRows_;
}

uint32 DenseLabelWiseStatisticsMatrix::getNumCols() const {
    return numCols_;
}

void DenseLabelWiseStatisticsMatrix::addToRow(uint32 row, gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd) {
    uint32 offset = row * numCols_;

    for (uint32 i = 0; i < numCols_; i++) {
        uint32 index = offset + i;
        gradients_[index] += gradientsBegin[i];
        hessians_[index] += hessiansBegin[i];
    }
}
