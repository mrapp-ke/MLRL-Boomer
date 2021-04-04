#include "seco/data/vector_dense_confusion_matrices.hpp"
#include "seco/heuristics/confusion_matrices.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


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

}
