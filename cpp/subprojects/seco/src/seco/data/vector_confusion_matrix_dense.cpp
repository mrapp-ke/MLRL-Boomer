#include "seco/data/vector_confusion_matrix_dense.hpp"
#include "common/data/arrays.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include <cstdlib>


namespace seco {

    template<typename LabelIterator>
    static inline void addInternally(LabelIterator labelIterator, const VectorConstView<uint32>& majorityLabelIndices,
                                     DenseCoverageMatrix::value_const_iterator coverageIterator, float64 weight,
                                     ConfusionMatrix* confusionMatrices, uint32 numLabels) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                             majorityLabelIndices.cend());

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = confusionMatrices[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements)
        : DenseConfusionMatrixVector(numElements, false) {

    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : array_(init ? (ConfusionMatrix*) calloc(numElements, sizeof(ConfusionMatrix))
                      : (ConfusionMatrix*) malloc(numElements * sizeof(ConfusionMatrix))),
          numElements_(numElements) {

    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other)
        : DenseConfusionMatrixVector(other.numElements_) {
        copyArray(other.array_, array_, numElements_);
    }

    DenseConfusionMatrixVector::~DenseConfusionMatrixVector() {
        free(array_);
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::begin() {
        return array_;
    }

    DenseConfusionMatrixVector::iterator DenseConfusionMatrixVector::end() {
        return &array_[numElements_];
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::cbegin() const {
        return array_;
    }

    DenseConfusionMatrixVector::const_iterator DenseConfusionMatrixVector::cend() const {
        return &array_[numElements_];
    }

    uint32 DenseConfusionMatrixVector::getNumElements() const {
        return numElements_;
    }

    void DenseConfusionMatrixVector::clear() {
        setArrayToZeros(array_, numElements_);
    }

    void DenseConfusionMatrixVector::add(const_iterator begin, const_iterator end) {
        for (uint32 i = 0; i < numElements_; i++) {
            array_[i] += begin[i];
        }
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                         const VectorConstView<uint32>& majorityLabelIndices,
                                         const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        addInternally(labelMatrix.row_values_cbegin(exampleIndex), majorityLabelIndices,
                      coverageMatrix.row_values_cbegin(exampleIndex), weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                         const VectorConstView<uint32>& majorityLabelIndices,
                                         const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        auto labelIterator = make_binary_forward_iterator(labelMatrix.row_indices_cbegin(exampleIndex),
                                                          labelMatrix.row_indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndices, coverageMatrix.row_values_cbegin(exampleIndex), weight,
                      array_, numElements_);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex,
                                                 const CContiguousConstView<const uint8>& labelMatrix,
                                                 const VectorConstView<uint32>& majorityLabelIndices,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        addInternally(labelMatrix.row_values_cbegin(exampleIndex), majorityLabelIndices,
                      coverageMatrix.row_values_cbegin(exampleIndex), weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                 const VectorConstView<uint32>& majorityLabelIndices,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        auto labelIterator = make_binary_forward_iterator(labelMatrix.row_indices_cbegin(exampleIndex),
                                                          labelMatrix.row_indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndices, coverageMatrix.row_values_cbegin(exampleIndex), weight,
                      array_, numElements_);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex,
                                                 const CContiguousConstView<const uint8>& labelMatrix,
                                                 const VectorConstView<uint32>& majorityLabelIndices,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                             majorityLabelIndices.cend());
        typename DenseCoverageMatrix::value_const_iterator coverageIterator =
            coverageMatrix.row_values_cbegin(exampleIndex);
        CContiguousConstView<const uint8>::value_const_iterator labelIterator =
            labelMatrix.row_values_cbegin(exampleIndex);
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            uint32 coverage = coverageIterator[index];

            if (coverage == 0) {
                bool trueLabel = labelIterator[index];
                std::advance(majorityIterator, index - previousIndex);
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = array_[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
                previousIndex = index;
            }
        }
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                 const VectorConstView<uint32>& majorityLabelIndices,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndices.cbegin(),
                                                             majorityLabelIndices.cend());
        typename DenseCoverageMatrix::value_const_iterator coverageIterator =
            coverageMatrix.row_values_cbegin(exampleIndex);
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.row_indices_cbegin(exampleIndex);
        BinaryCsrConstView::index_const_iterator labelIndicesEnd = labelMatrix.row_indices_cend(exampleIndex);
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            uint32 coverage = coverageIterator[index];

            if (coverage == 0) {
                labelIndexIterator = std::lower_bound(labelIndexIterator, labelIndicesEnd, index);
                bool trueLabel = labelIndexIterator != labelIndicesEnd && *labelIndexIterator == index;
                std::advance(majorityIterator, index - previousIndex);
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = array_[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
                previousIndex = index;
            }
        }
    }

    void DenseConfusionMatrixVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                const CompleteIndexVector& firstIndices, const_iterator secondBegin,
                                                const_iterator secondEnd) {
        setArrayToDifference(array_, firstBegin, secondBegin, numElements_);
    }

    void DenseConfusionMatrixVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                const PartialIndexVector& firstIndices, const_iterator secondBegin,
                                                const_iterator secondEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setArrayToDifference(array_, firstBegin, secondBegin, indexIterator, numElements_);
    }

}
