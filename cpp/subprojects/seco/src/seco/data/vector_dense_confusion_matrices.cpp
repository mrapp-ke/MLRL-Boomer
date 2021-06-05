#include "seco/data/vector_dense_confusion_matrices.hpp"
#include "common/data/arrays.hpp"
#include "confusion_matrices.hpp"
#include <cstdlib>

#define NUM_CONFUSION_MATRIX_ELEMENTS 4


namespace seco {

    template<typename LabelMatrix>
    static inline void addInternally(DenseConfusionMatrixVector& confusionMatrixVector, uint32 exampleIndex,
                                     const LabelMatrix& labelMatrix, const BinarySparseArrayVector& majorityLabelVector,
                                     const DenseWeightMatrix& weightMatrix, float64 weight) {
        auto majorityIterator = make_index_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                            majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        typename LabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numElements = confusionMatrixVector.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            float64 labelWeight = weightIterator[i];

            if (labelWeight > 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                DenseConfusionMatrixVector::iterator iterator = confusionMatrixVector.confusion_matrix_begin(i);
                uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                iterator[element] += (labelWeight * weight);
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename LabelMatrix>
    static inline void addToSubsetInternally(DenseConfusionMatrixVector& confusionMatrixVector, uint32 exampleIndex,
                                             const LabelMatrix& labelMatrix,
                                             const BinarySparseArrayVector& majorityLabelVector,
                                             const DenseWeightMatrix& weightMatrix, float64 weight) {
        auto majorityIterator = make_index_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                            majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        typename LabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numElements = confusionMatrixVector.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            float64 labelWeight = weightIterator[i];

            if (labelWeight > 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                DenseConfusionMatrixVector::iterator iterator = confusionMatrixVector.confusion_matrix_begin(i);
                uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                iterator[element] += (labelWeight * weight);
            }

            labelIterator++;
            majorityIterator++;
        }
    }

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

    void DenseConfusionMatrixVector::clear() {
        setArrayToZeros(array_, numElements_ * NUM_CONFUSION_MATRIX_ELEMENTS);
    }

    void DenseConfusionMatrixVector::add(const_iterator begin, const_iterator end) {
        uint32 numElements = numElements_ * NUM_CONFUSION_MATRIX_ELEMENTS;

        for (uint32 i = 0; i < numElements; i++) {
            array_[i] += begin[i];
        }
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                         const BinarySparseArrayVector& majorityLabelVector,
                                         const DenseWeightMatrix& weightMatrix, float64 weight) {
        addInternally<CContiguousLabelMatrix>(*this, exampleIndex, labelMatrix, majorityLabelVector, weightMatrix,
                                              weight);
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                         const BinarySparseArrayVector& majorityLabelVector,
                                         const DenseWeightMatrix& weightMatrix, float64 weight) {
        addInternally<CsrLabelMatrix>(*this, exampleIndex, labelMatrix, majorityLabelVector, weightMatrix, weight);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix, const FullIndexVector& indices,
                                                 float64 weight) {
        addToSubsetInternally<CContiguousLabelMatrix>(*this, exampleIndex, labelMatrix, majorityLabelVector,
                                                      weightMatrix, weight);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix, const FullIndexVector& indices,
                                                 float64 weight) {
        addToSubsetInternally<CsrLabelMatrix>(*this, exampleIndex, labelMatrix, majorityLabelVector, weightMatrix,
                                              weight);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_index_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                            majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            float64 labelWeight = weightIterator[index];

            if (labelWeight > 0) {
                bool trueLabel = labelIterator[index];
                std::advance(majorityIterator, index - previousIndex);
                bool majorityLabel = *majorityIterator;
                iterator confusionMatrixIterator = this->confusion_matrix_begin(i);
                uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                confusionMatrixIterator[element] += (labelWeight * weight);
                previousIndex = index;
            }
        }
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                 const BinarySparseArrayVector& majorityLabelVector,
                                                 const DenseWeightMatrix& weightMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_index_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                            majorityLabelVector.indices_cend());
        typename DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            float64 labelWeight = weightIterator[index];

            if (labelWeight > 0) {
                std::advance(labelIterator, index - previousIndex);
                bool trueLabel = *labelIterator;
                std::advance(majorityIterator, index - previousIndex);
                bool majorityLabel = *majorityIterator;
                iterator confusionMatrixIterator = this->confusion_matrix_begin(i);
                uint32 element = getConfusionMatrixElement(trueLabel, majorityLabel);
                confusionMatrixIterator[element] += (labelWeight * weight);
                previousIndex = index;
            }
        }
    }

}
