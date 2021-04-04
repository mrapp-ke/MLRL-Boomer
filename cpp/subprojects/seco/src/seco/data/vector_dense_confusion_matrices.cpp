#include "seco/data/vector_dense_confusion_matrices.hpp"
#include "common/data/arrays.hpp"
#include "confusion_matrices.hpp"
#include <cstdlib>

#define NUM_CONFUSION_MATRIX_ELEMENTS 4


namespace seco {

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements)
        : DenseConfusionMatrixVector(numElements, false) {

    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : array_(init ? (float64*) calloc(numElements * NUM_CONFUSION_MATRIX_ELEMENTS, sizeof(float64))
                      : (float64*) malloc(numElements * NUM_CONFUSION_MATRIX_ELEMENTS * sizeof(float64))),
          numElements_(numElements) {

    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other)
        : DenseConfusionMatrixVector(other.numElements_) {
        copyArray(other.array_, array_, numElements_ * NUM_CONFUSION_MATRIX_ELEMENTS);
    }

    DenseConfusionMatrixVector::~DenseConfusionMatrixVector() {
        free(array_);
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::begin() {
        return array_;
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::end() {
        return &array_[numElements_ * NUM_CONFUSION_MATRIX_ELEMENTS];
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::cbegin() const {
        return array_;
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::cend() const {
        return &array_[numElements_ * NUM_CONFUSION_MATRIX_ELEMENTS];
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::confusion_matrix_begin(uint32 pos) {
        return &array_[pos * NUM_CONFUSION_MATRIX_ELEMENTS];
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::confusion_matrix_end(uint32 pos) {
        return &array_[(pos + 1) * NUM_CONFUSION_MATRIX_ELEMENTS];
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::confusion_matrix_cbegin(uint32 pos) const {
        return &array_[pos * NUM_CONFUSION_MATRIX_ELEMENTS];
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::confusion_matrix_cend(uint32 pos) const {
        return &array_[(pos + 1) * NUM_CONFUSION_MATRIX_ELEMENTS];
    }

    uint32 DenseConfusionMatrixVector::getNumElements() const {
        return numElements_;
    }

    void DenseConfusionMatrixVector::setAllToZero() {
        setArrayToZeros(array_, numElements_ * NUM_CONFUSION_MATRIX_ELEMENTS);
    }

    void DenseConfusionMatrixVector::add(const_iterator begin, const_iterator end) {
        uint32 numElements = numElements_ * NUM_CONFUSION_MATRIX_ELEMENTS;

        for (uint32 i = 0; i < numElements; i++) {
            array_[i] += begin[i];
        }
    }

    void DenseConfusionMatrixVector::add(uint32 row, const IRandomAccessLabelMatrix& labelMatrix,
                                         const DenseVector<uint8>& majorityLabelVector,
                                         const DenseWeightMatrix& weightMatrix, float64 weight) {
        DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVector.cbegin();
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(row);

        for (uint32 i = 0; i < numElements_; i++) {
            iterator confusionMatrixIterator = this->confusion_matrix_begin(i);
            float64 labelWeight = weightIterator[i];

            if (labelWeight > 0) {
                uint8 trueLabel = labelMatrix.getValue(row, i);
                uint8 majorityLabel = majorityIterator[i];
                uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                confusionMatrixIterator[element] += (labelWeight * weight);
            }
        }
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 row, const IRandomAccessLabelMatrix& labelMatrix,
                                                 const DenseVector<uint8>& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 FullIndexVector::const_iterator indicesBegin,
                                                 FullIndexVector::const_iterator indicesEnd, float64 weight) {
        DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVector.cbegin();
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(row);

        for (uint32 i = 0; i < numElements_; i++) {
            iterator confusionMatrixIterator = this->confusion_matrix_begin(i);
            float64 labelWeight = weightIterator[i];

            if (labelWeight > 0) {
                uint8 trueLabel = labelMatrix.getValue(row, i);
                uint8 majorityLabel = majorityIterator[i];
                uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                confusionMatrixIterator[element] += (labelWeight * weight);
            }
        }
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 row, const IRandomAccessLabelMatrix& labelMatrix,
                                                 const DenseVector<uint8>& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 PartialIndexVector::const_iterator indicesBegin,
                                                 PartialIndexVector::const_iterator indicesEnd, float64 weight) {
        DenseVector<uint8>::const_iterator majorityIterator = majorityLabelVector.cbegin();
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(row);
        uint32 numElements = indicesEnd - indicesBegin;

        for (uint32 i = 0; i < numElements; i++) {
            iterator confusionMatrixIterator = this->confusion_matrix_begin(i);
            uint32 index = indicesBegin[i];
            float64 labelWeight = weightIterator[index];

            if (labelWeight > 0) {
                uint8 trueLabel = labelMatrix.getValue(row, index);
                uint8 majorityLabel = majorityIterator[index];
                uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                confusionMatrixIterator[element] += (labelWeight * weight);
            }
        }
    }

}
