#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/util/xsimd.hpp"

namespace seco {

    template<typename StatisticType, typename LabelIterator>
    static inline void addInternally(typename View<ConfusionMatrix<StatisticType>>::iterator statisticIterator,
                                     LabelIterator labelIterator,
                                     View<uint32>::const_iterator majorityLabelIndicesBegin,
                                     View<uint32>::const_iterator majorityLabelIndicesEnd,
                                     DenseCoverageMatrix::value_const_iterator coverageIterator, StatisticType weight,
                                     uint32 numLabels) {
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix<StatisticType>& confusionMatrix = statisticIterator[i];
                StatisticType& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element += weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename StatisticType, typename LabelIterator>
    static inline void removeInternally(typename View<ConfusionMatrix<StatisticType>>::iterator statisticIterator,
                                        LabelIterator labelIterator,
                                        View<uint32>::const_iterator majorityLabelIndicesBegin,
                                        View<uint32>::const_iterator majorityLabelIndicesEnd,
                                        DenseCoverageMatrix::value_const_iterator coverageIterator,
                                        StatisticType weight, uint32 numLabels) {
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                ConfusionMatrix<StatisticType>& confusionMatrix = statisticIterator[i];
                StatisticType& element = confusionMatrix.getElement(trueLabel, majorityLabel);
                element -= weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename StatisticType, typename ArrayOperations>
    DenseConfusionMatrixVector<StatisticType, ArrayOperations>::DenseConfusionMatrixVector(uint32 numElements,
                                                                                           bool init)
        : ClearableViewDecorator<DenseVectorDecorator<DenseConfusionMatrixVectorView<StatisticType>>>(
            DenseConfusionMatrixVectorView<StatisticType>(numElements, init)) {}

    template<typename StatisticType, typename ArrayOperations>
    DenseConfusionMatrixVector<StatisticType, ArrayOperations>::DenseConfusionMatrixVector(
      const DenseConfusionMatrixVector<StatisticType, ArrayOperations>& other)
        : DenseConfusionMatrixVector(other.getNumElements()) {
        ArrayOperations::copy(other.cbegin(), this->begin(), this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::add(
      const DenseConfusionMatrixVectorView<StatisticType>& other) {
        ArrayOperations::add(this->begin(), other.cbegin(), this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::add(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      StatisticType weight) {
        addInternally(this->begin(), view.labelMatrix.values_cbegin(row), view.majorityLabelVector.cbegin(),
                      view.majorityLabelVector.cend(), view.coverageMatrix.values_cbegin(row), weight,
                      this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::add(
      const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row, StatisticType weight) {
        auto labelIterator =
          createBinarySparseForwardIterator(view.labelMatrix.indices_cbegin(row), view.labelMatrix.indices_cend(row));
        addInternally(this->begin(), labelIterator, view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend(),
                      view.coverageMatrix.values_cbegin(row), weight, this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::remove(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      StatisticType weight) {
        removeInternally(this->begin(), view.labelMatrix.values_cbegin(row), view.majorityLabelVector.cbegin(),
                         view.majorityLabelVector.cend(), view.coverageMatrix.values_cbegin(row), weight,
                         this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::remove(
      const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row, StatisticType weight) {
        auto labelIterator =
          createBinarySparseForwardIterator(view.labelMatrix.indices_cbegin(row), view.labelMatrix.indices_cend(row));
        removeInternally(this->begin(), labelIterator, view.majorityLabelVector.cbegin(),
                         view.majorityLabelVector.cend(), view.coverageMatrix.values_cbegin(row), weight,
                         this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      const CompleteIndexVector& indices, StatisticType weight) {
        addInternally(this->begin(), view.labelMatrix.values_cbegin(row), view.majorityLabelVector.cbegin(),
                      view.majorityLabelVector.cend(), view.coverageMatrix.values_cbegin(row), weight,
                      this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        auto labelIterator =
          createBinarySparseForwardIterator(view.labelMatrix.indices_cbegin(row), view.labelMatrix.indices_cend(row));
        addInternally(this->begin(), labelIterator, view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend(),
                      view.coverageMatrix.values_cbegin(row), weight, this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      const PartialIndexVector& indices, StatisticType weight) {
        auto majorityIterator =
          createBinarySparseForwardIterator(view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend());
        typename DenseCoverageMatrix::value_const_iterator coverageIterator = view.coverageMatrix.values_cbegin(row);
        CContiguousView<const uint8>::value_const_iterator labelIterator = view.labelMatrix.values_cbegin(row);
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

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        auto majorityIterator =
          createBinarySparseForwardIterator(view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend());
        typename DenseCoverageMatrix::value_const_iterator coverageIterator = view.coverageMatrix.values_cbegin(row);
        BinaryCsrView::index_const_iterator labelIndexIterator = view.labelMatrix.indices_cbegin(row);
        BinaryCsrView::index_const_iterator labelIndicesEnd = view.labelMatrix.indices_cend(row);
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

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::difference(
      const DenseConfusionMatrixVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseConfusionMatrixVectorView<StatisticType>& second) {
        ArrayOperations::difference(this->begin(), first.cbegin(), second.cbegin(), this->getNumElements());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseConfusionMatrixVector<StatisticType, ArrayOperations>::difference(
      const DenseConfusionMatrixVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseConfusionMatrixVectorView<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        ArrayOperations::difference(this->begin(), first.cbegin(), second.cbegin(), indexIterator,
                                    this->getNumElements());
    }

    template class DenseConfusionMatrixVector<uint32, SequentialArrayOperations>;
    template class DenseConfusionMatrixVector<float32, SequentialArrayOperations>;

#if SIMD_SUPPORT_ENABLED
    template class DenseConfusionMatrixVector<uint32, SimdArrayOperations>;
    template class DenseConfusionMatrixVector<float32, SimdArrayOperations>;
#endif
}
