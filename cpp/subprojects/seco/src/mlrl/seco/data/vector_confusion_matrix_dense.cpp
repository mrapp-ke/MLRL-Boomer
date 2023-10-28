#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"

#include "mlrl/common/iterator/binary_forward_iterator.hpp"
#include "mlrl/common/util/memory.hpp"
#include "mlrl/common/util/view_functions.hpp"

#include <algorithm>

namespace seco {

    template<typename LabelIterator>
    static inline void addInternally(LabelIterator labelIterator,
                                     View<uint32>::const_iterator majorityLabelIndicesBegin,
                                     View<uint32>::const_iterator majorityLabelIndicesEnd,
                                     DenseCoverageMatrix::value_const_iterator coverageIterator, float64 weight,
                                     ConfusionMatrix* confusionMatrices, uint32 numLabels) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

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

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : array_(allocateMemory<ConfusionMatrix>(numElements, init)), numElements_(numElements) {}

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other)
        : DenseConfusionMatrixVector(other.numElements_) {
        copyView(other.array_, array_, numElements_);
    }

    DenseConfusionMatrixVector::~DenseConfusionMatrixVector() {
        freeMemory(array_);
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
        setViewToZeros(array_, numElements_);
    }

    void DenseConfusionMatrixVector::add(const_iterator begin, const_iterator end) {
        for (uint32 i = 0; i < numElements_; i++) {
            array_[i] += begin[i];
        }
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                         View<uint32>::const_iterator majorityLabelIndicesBegin,
                                         View<uint32>::const_iterator majorityLabelIndicesEnd,
                                         const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        addInternally(labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                         View<uint32>::const_iterator majorityLabelIndicesBegin,
                                         View<uint32>::const_iterator majorityLabelIndicesEnd,
                                         const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                          labelMatrix.indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::remove(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                            View<uint32>::const_iterator majorityLabelIndicesBegin,
                                            View<uint32>::const_iterator majorityLabelIndicesEnd,
                                            const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        addInternally(labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), -weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::remove(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                            View<uint32>::const_iterator majorityLabelIndicesBegin,
                                            View<uint32>::const_iterator majorityLabelIndicesEnd,
                                            const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                          labelMatrix.indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), -weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex,
                                                 const CContiguousConstView<const uint8>& labelMatrix,
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        addInternally(labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                          labelMatrix.indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, array_, numElements_);
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex,
                                                 const CContiguousConstView<const uint8>& labelMatrix,
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
        typename DenseCoverageMatrix::value_const_iterator coverageIterator =
          coverageMatrix.values_cbegin(exampleIndex);
        CContiguousConstView<const uint8>::value_const_iterator labelIterator = labelMatrix.values_cbegin(exampleIndex);
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
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = make_binary_forward_iterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
        typename DenseCoverageMatrix::value_const_iterator coverageIterator =
          coverageMatrix.values_cbegin(exampleIndex);
        BinaryCsrConstView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        BinaryCsrConstView::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
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
        setViewToDifference(array_, firstBegin, secondBegin, numElements_);
    }

    void DenseConfusionMatrixVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                const PartialIndexVector& firstIndices, const_iterator secondBegin,
                                                const_iterator secondEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setViewToDifference(array_, firstBegin, secondBegin, indexIterator, numElements_);
    }

}
