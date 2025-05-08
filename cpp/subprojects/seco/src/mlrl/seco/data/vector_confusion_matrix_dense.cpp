#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"

#include <algorithm>

namespace seco {

    template<typename StatisticType, typename LabelIterator>
    static inline void addInternally(DenseConfusionMatrixVector<StatisticType>& vector, LabelIterator labelIterator,
                                     View<uint32>::const_iterator majorityLabelIndicesBegin,
                                     View<uint32>::const_iterator majorityLabelIndicesEnd,
                                     DenseCoverageMatrix::value_const_iterator coverageIterator, StatisticType weight,
                                     uint32 numLabels) {
        typename DenseConfusionMatrixVector<StatisticType>::iterator iterator = vector.begin();
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix<StatisticType>& confusionMatrix = iterator[i];
                StatisticType& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename StatisticType, typename LabelIterator>
    static inline void removeInternally(DenseConfusionMatrixVector<StatisticType>& vector, LabelIterator labelIterator,
                                        View<uint32>::const_iterator majorityLabelIndicesBegin,
                                        View<uint32>::const_iterator majorityLabelIndicesEnd,
                                        DenseCoverageMatrix::value_const_iterator coverageIterator,
                                        StatisticType weight, uint32 numLabels) {
        typename DenseConfusionMatrixVector<StatisticType>::iterator iterator = vector.begin();
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix<StatisticType>& confusionMatrix = iterator[i];
                StatisticType& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element -= weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename StatisticType>
    DenseConfusionMatrixVector<StatisticType>::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<ConfusionMatrix<StatisticType>>>>(
            AllocatedVector<ConfusionMatrix<StatisticType>>(numElements, init)) {}

    template<typename StatisticType>
    DenseConfusionMatrixVector<StatisticType>::DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other)
        : DenseConfusionMatrixVector(other.getNumElements()) {
        util::copyView(other.cbegin(), this->begin(), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::add(
      typename View<ConfusionMatrix<StatisticType>>::const_iterator begin,
      typename View<ConfusionMatrix<StatisticType>>::const_iterator end) {
        util::addToView(this->begin(), begin, this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::add(uint32 exampleIndex,
                                                        const CContiguousView<const uint8>& labelMatrix,
                                                        View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                        View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                        const DenseCoverageMatrix& coverageMatrix,
                                                        StatisticType weight) {
        addInternally(*this, labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin,
                      majorityLabelIndicesEnd, coverageMatrix.values_cbegin(exampleIndex), weight,
                      this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::add(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                        View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                        View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                        const DenseCoverageMatrix& coverageMatrix,
                                                        StatisticType weight) {
        auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                               labelMatrix.indices_cend(exampleIndex));
        addInternally(*this, labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::remove(uint32 exampleIndex,
                                                           const CContiguousView<const uint8>& labelMatrix,
                                                           View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                           View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                           const DenseCoverageMatrix& coverageMatrix,
                                                           StatisticType weight) {
        removeInternally(*this, labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin,
                         majorityLabelIndicesEnd, coverageMatrix.values_cbegin(exampleIndex), weight,
                         this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::remove(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                           View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                           View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                           const DenseCoverageMatrix& coverageMatrix,
                                                           StatisticType weight) {
        auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                               labelMatrix.indices_cend(exampleIndex));
        removeInternally(*this, labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                         coverageMatrix.values_cbegin(exampleIndex), weight, this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::addToSubset(
      uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
      View<uint32>::const_iterator majorityLabelIndicesBegin, View<uint32>::const_iterator majorityLabelIndicesEnd,
      const DenseCoverageMatrix& coverageMatrix, const CompleteIndexVector& indices, StatisticType weight) {
        addInternally(*this, labelMatrix.values_cbegin(exampleIndex), majorityLabelIndicesBegin,
                      majorityLabelIndicesEnd, coverageMatrix.values_cbegin(exampleIndex), weight,
                      this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::addToSubset(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                                View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                                View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                                const DenseCoverageMatrix& coverageMatrix,
                                                                const CompleteIndexVector& indices,
                                                                StatisticType weight) {
        auto labelIterator = createBinarySparseForwardIterator(labelMatrix.indices_cbegin(exampleIndex),
                                                               labelMatrix.indices_cend(exampleIndex));
        addInternally(*this, labelIterator, majorityLabelIndicesBegin, majorityLabelIndicesEnd,
                      coverageMatrix.values_cbegin(exampleIndex), weight, this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::addToSubset(
      uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
      View<uint32>::const_iterator majorityLabelIndicesBegin, View<uint32>::const_iterator majorityLabelIndicesEnd,
      const DenseCoverageMatrix& coverageMatrix, const PartialIndexVector& indices, StatisticType weight) {
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
                ConfusionMatrix<StatisticType>& confusionMatrix = this->begin()[i];
                StatisticType& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
                previousIndex = index;
            }
        }
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::addToSubset(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                                                                View<uint32>::const_iterator majorityLabelIndicesBegin,
                                                                View<uint32>::const_iterator majorityLabelIndicesEnd,
                                                                const DenseCoverageMatrix& coverageMatrix,
                                                                const PartialIndexVector& indices,
                                                                StatisticType weight) {
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
                ConfusionMatrix<StatisticType>& confusionMatrix = this->begin()[i];
                StatisticType& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
                previousIndex = index;
            }
        }
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::difference(
      typename View<ConfusionMatrix<StatisticType>>::const_iterator firstBegin,
      typename View<ConfusionMatrix<StatisticType>>::const_iterator firstEnd, const CompleteIndexVector& firstIndices,
      typename View<ConfusionMatrix<StatisticType>>::const_iterator secondBegin,
      typename View<ConfusionMatrix<StatisticType>>::const_iterator secondEnd) {
        util::setViewToDifference(this->begin(), firstBegin, secondBegin, this->getNumElements());
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVector<StatisticType>::difference(
      typename View<ConfusionMatrix<StatisticType>>::const_iterator firstBegin,
      typename View<ConfusionMatrix<StatisticType>>::const_iterator firstEnd, const PartialIndexVector& firstIndices,
      typename View<ConfusionMatrix<StatisticType>>::const_iterator secondBegin,
      typename View<ConfusionMatrix<StatisticType>>::const_iterator secondEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        util::setViewToDifference(this->begin(), firstBegin, secondBegin, indexIterator, this->getNumElements());
    }

    template class DenseConfusionMatrixVector<uint32>;
    template class DenseConfusionMatrixVector<float32>;
}
