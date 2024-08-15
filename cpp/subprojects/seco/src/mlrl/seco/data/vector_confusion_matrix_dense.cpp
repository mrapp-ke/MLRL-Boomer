#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"

#include <algorithm>

namespace seco {

    template<typename LabelIterator>
    static inline void addInternally(LabelIterator labelIterator,
                                     View<uint32>::const_iterator majorityLabelIndicesBegin,
                                     View<uint32>::const_iterator majorityLabelIndicesEnd,
                                     DenseCoverageMatrix::value_const_iterator coverageIterator, float64 weight,
                                     DenseConfusionMatrixVector::iterator iterator, uint32 numLabels) {
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix& confusionMatrix = iterator[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<ConfusionMatrix>>>(
            AllocatedVector<ConfusionMatrix>(numElements, init)) {}

    DenseConfusionMatrixVector::DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other)
        : DenseConfusionMatrixVector(other.getNumElements()) {
        util::copyView(other.cbegin(), this->begin(), this->getNumElements());
    }

    void DenseConfusionMatrixVector::add(const_iterator begin, const_iterator end) {
        util::addToView(this->begin(), begin, this->getNumElements());
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                         View<uint32>::const_iterator majorityLabelIndicesBegin,
                                         View<uint32>::const_iterator majorityLabelIndicesEnd,
                                         const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        addInternally(labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, this->begin(), this->getNumElements());
    }

    void DenseConfusionMatrixVector::add(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                         View<uint32>::const_iterator majorityLabelIndicesBegin,
                                         View<uint32>::const_iterator majorityLabelIndicesEnd,
                                         const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                               labelMatrix.indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, this->begin(), this->getNumElements());
    }

    void DenseConfusionMatrixVector::remove(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                            View<uint32>::const_iterator majorityLabelIndicesBegin,
                                            View<uint32>::const_iterator majorityLabelIndicesEnd,
                                            const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        addInternally(labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), -weight, this->begin(), this->getNumElements());
    }

    void DenseConfusionMatrixVector::remove(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                            View<uint32>::const_iterator majorityLabelIndicesBegin,
                                            View<uint32>::const_iterator majorityLabelIndicesEnd,
                                            const DenseCoverageMatrix& coverageMatrix, float64 weight) {
        auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                               labelMatrix.indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), -weight, this->begin(), this->getNumElements());
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        addInternally(labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, this->begin(), this->getNumElements());
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const CompleteIndexVector& indices, float64 weight) {
        auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                               labelMatrix.indices_cend(exampleIndex));
        addInternally(labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, this->begin(), this->getNumElements());
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
        typename DenseCoverageMatrix::value_const_iterator coverageIterator =
          coverageMatrix.values_cbegin(exampleIndex);
        CContiguousView<const uint8>::value_const_iterator labelIterator = labelMatrix.values_cbegin(exampleIndex);
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
                ConfusionMatrix& confusionMatrix = this->begin()[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
                previousIndex = index;
            }
        }
    }

    void DenseConfusionMatrixVector::addToSubset(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                 View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                 View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                 const DenseCoverageMatrix& coverageMatrix,
                                                 const PartialIndexVector& indices, float64 weight) {
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);
        typename DenseCoverageMatrix::value_const_iterator coverageIterator =
          coverageMatrix.values_cbegin(exampleIndex);
        BinaryCsrView::index_const_iterator labelIndexIterator = labelMatrix.indices_cbegin(exampleIndex);
        BinaryCsrView::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
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
                ConfusionMatrix& confusionMatrix = this->begin()[i];
                float64& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
                previousIndex = index;
            }
        }
    }

    void DenseConfusionMatrixVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                const CompleteIndexVector& firstIndices, const_iterator secondBegin,
                                                const_iterator secondEnd) {
        util::setViewToDifference(this->begin(), firstBegin, secondBegin, this->getNumElements());
    }

    void DenseConfusionMatrixVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                const PartialIndexVector& firstIndices, const_iterator secondBegin,
                                                const_iterator secondEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        util::setViewToDifference(this->begin(), firstBegin, secondBegin, indexIterator, this->getNumElements());
    }

}
