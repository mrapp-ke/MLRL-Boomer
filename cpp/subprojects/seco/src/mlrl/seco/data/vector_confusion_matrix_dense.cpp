#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"

namespace seco {

    template<typename StatisticType>
    DenseConfusionMatrixVectorView<StatisticType>::DenseConfusionMatrixVectorView(uint32 numElements, bool init)
        : CompositeVector<CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>,
                          CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>>(
            CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
              AllocatedVector<StatisticType>(numElements, init), AllocatedVector<StatisticType>(numElements, init)),
            CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
              AllocatedVector<StatisticType>(numElements, init), AllocatedVector<StatisticType>(numElements, init))) {}

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::in_cbegin() const {
        return this->firstView.firstView.cbegin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::in_cend() const {
        return this->firstView.firstView.cend();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::in_begin() {
        return this->firstView.firstView.begin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::in_end() {
        return this->firstView.firstView.end();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::ip_cbegin() const {
        return this->firstView.secondView.cbegin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::ip_cend() const {
        return this->firstView.secondView.cend();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::ip_begin() {
        return this->firstView.secondView.begin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::ip_end() {
        return this->firstView.secondView.end();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::rn_cbegin() const {
        return this->secondView.firstView.cbegin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::rn_cend() const {
        return this->secondView.firstView.cend();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::rn_begin() {
        return this->secondView.firstView.begin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::rn_end() {
        return this->secondView.firstView.end();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::rp_cbegin() const {
        return this->secondView.secondView.cbegin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVectorView<StatisticType>::rp_cend() const {
        return this->secondView.secondView.cend();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::rp_begin() {
        return this->secondView.secondView.begin();
    }

    template<typename StatisticType>
    typename View<StatisticType>::iterator DenseConfusionMatrixVectorView<StatisticType>::rp_end() {
        return this->secondView.secondView.end();
    }

    template<typename StatisticType>
    uint32 DenseConfusionMatrixVectorView<StatisticType>::getNumElements() const {
        return this->firstView.firstView.numElements;
    }

    template<typename StatisticType>
    void DenseConfusionMatrixVectorView<StatisticType>::clear() {
        this->firstView.firstView.clear();
        this->firstView.secondView.clear();
        this->secondView.firstView.clear();
        this->secondView.secondView.clear();
    }

    template class DenseConfusionMatrixVectorView<uint32>;
    template class DenseConfusionMatrixVectorView<float32>;

    template<typename StatisticType>
    static inline const StatisticType& getElement(bool trueLabel, bool majorityLabel, const StatisticType& in,
                                                  const StatisticType& ip, const StatisticType& rn,
                                                  const StatisticType& rp) {
        if (trueLabel) {
            return majorityLabel ? rn : rp;
        } else {
            return majorityLabel ? in : ip;
        }
    }

    template<typename StatisticType>
    static inline StatisticType& getElement(bool trueLabel, bool majorityLabel, StatisticType& in, StatisticType& ip,
                                            StatisticType& rn, StatisticType& rp) {
        if (trueLabel) {
            return majorityLabel ? rn : rp;
        } else {
            return majorityLabel ? in : ip;
        }
    }

    template<typename StatisticType, typename LabelIterator>
    static inline void addInternally(
      typename View<StatisticType>::iterator in, typename View<StatisticType>::iterator ip,
      typename View<StatisticType>::iterator rn, typename View<StatisticType>::iterator rp, LabelIterator labelIterator,
      View<uint32>::const_iterator majorityLabelIndicesBegin, View<uint32>::const_iterator majorityLabelIndicesEnd,
      DenseCoverageMatrix::value_const_iterator coverageIterator, StatisticType weight, uint32 numLabels) {
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                StatisticType& element = getElement(trueLabel, majorityLabel, in[i], ip[i], rn[i], rp[i]);
                element += weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename StatisticType, typename LabelIterator>
    static inline void removeInternally(
      typename View<StatisticType>::iterator in, typename View<StatisticType>::iterator ip,
      typename View<StatisticType>::iterator rn, typename View<StatisticType>::iterator rp, LabelIterator labelIterator,
      View<uint32>::const_iterator majorityLabelIndicesBegin, View<uint32>::const_iterator majorityLabelIndicesEnd,
      DenseCoverageMatrix::value_const_iterator coverageIterator, StatisticType weight, uint32 numLabels) {
        auto majorityIterator = createBinarySparseForwardIterator(majorityLabelIndicesBegin, majorityLabelIndicesEnd);

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 coverage = coverageIterator[i];

            if (coverage == 0) {
                bool trueLabel = *labelIterator;
                bool majorityLabel = *majorityIterator;
                StatisticType& element = getElement(trueLabel, majorityLabel, in[i], ip[i], rn[i], rp[i]);
                element -= weight;
            }

            labelIterator++;
            majorityIterator++;
        }
    }

    template<typename StatisticType, typename VectorMath>
    DenseConfusionMatrixVector<StatisticType, VectorMath>::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : ClearableViewDecorator<ViewDecorator<DenseConfusionMatrixVectorView<StatisticType>>>(
            DenseConfusionMatrixVectorView<StatisticType>(numElements, init)) {}

    template<typename StatisticType, typename VectorMath>
    DenseConfusionMatrixVector<StatisticType, VectorMath>::DenseConfusionMatrixVector(
      const DenseConfusionMatrixVector<StatisticType, VectorMath>& other)
        : DenseConfusionMatrixVector(other.getNumElements()) {
        uint32 numElements = this->getNumElements();
        SequentialVectorMath::copy(other.in_cbegin(), this->in_begin(), numElements);
        SequentialVectorMath::copy(other.ip_cbegin(), this->ip_begin(), numElements);
        SequentialVectorMath::copy(other.rn_cbegin(), this->rn_begin(), numElements);
        SequentialVectorMath::copy(other.rp_cbegin(), this->rp_begin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::add(
      const DenseConfusionMatrixVectorView<StatisticType>& other) {
        uint32 numElements = this->getNumElements();
        SequentialVectorMath::add(this->in_begin(), other.in_cbegin(), numElements);
        SequentialVectorMath::add(this->ip_begin(), other.ip_cbegin(), numElements);
        SequentialVectorMath::add(this->rn_begin(), other.rn_cbegin(), numElements);
        SequentialVectorMath::add(this->rp_begin(), other.rp_cbegin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::in_cbegin()
      const {
        return this->view.in_cbegin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::in_cend()
      const {
        return this->view.in_cend();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::in_begin() {
        return this->view.in_begin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::in_end() {
        return this->view.in_end();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::ip_cbegin()
      const {
        return this->view.ip_cbegin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::ip_cend()
      const {
        return this->view.ip_cend();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::ip_begin() {
        return this->view.ip_begin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::ip_end() {
        return this->view.ip_end();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rn_cbegin()
      const {
        return this->view.rn_cbegin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rn_cend()
      const {
        return this->view.rn_cend();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rn_begin() {
        return this->view.rn_begin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rn_end() {
        return this->view.rn_end();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rp_cbegin()
      const {
        return this->view.rp_cbegin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::const_iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rp_cend()
      const {
        return this->view.rp_cend();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rp_begin() {
        return this->view.rp_begin();
    }

    template<typename StatisticType, typename VectorMath>
    typename View<StatisticType>::iterator DenseConfusionMatrixVector<StatisticType, VectorMath>::rp_end() {
        return this->view.rp_end();
    }

    template<typename StatisticType, typename VectorMath>
    uint32 DenseConfusionMatrixVector<StatisticType, VectorMath>::getNumElements() const {
        return this->view.getNumElements();
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::add(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      StatisticType weight) {
        addInternally(this->in_begin(), this->ip_begin(), this->rn_begin(), this->rp_begin(),
                      view.labelMatrix.values_cbegin(row), view.majorityLabelVector.cbegin(),
                      view.majorityLabelVector.cend(), view.coverageMatrix.values_cbegin(row), weight,
                      this->getNumElements());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::add(
      const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row, StatisticType weight) {
        auto labelIterator =
          createBinarySparseForwardIterator(view.labelMatrix.indices_cbegin(row), view.labelMatrix.indices_cend(row));
        addInternally(this->in_begin(), this->ip_begin(), this->rn_begin(), this->rp_begin(), labelIterator,
                      view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend(),
                      view.coverageMatrix.values_cbegin(row), weight, this->getNumElements());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::remove(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      StatisticType weight) {
        removeInternally(this->in_begin(), this->ip_begin(), this->rn_begin(), this->rp_begin(),
                         view.labelMatrix.values_cbegin(row), view.majorityLabelVector.cbegin(),
                         view.majorityLabelVector.cend(), view.coverageMatrix.values_cbegin(row), weight,
                         this->getNumElements());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::remove(
      const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row, StatisticType weight) {
        auto labelIterator =
          createBinarySparseForwardIterator(view.labelMatrix.indices_cbegin(row), view.labelMatrix.indices_cend(row));
        removeInternally(this->in_begin(), this->ip_begin(), this->rn_begin(), this->rp_begin(), labelIterator,
                         view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend(),
                         view.coverageMatrix.values_cbegin(row), weight, this->getNumElements());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::addToSubset(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      const CompleteIndexVector& indices, StatisticType weight) {
        addInternally(this->in_begin(), this->ip_begin(), this->rn_begin(), this->rp_begin(),
                      view.labelMatrix.values_cbegin(row), view.majorityLabelVector.cbegin(),
                      view.majorityLabelVector.cend(), view.coverageMatrix.values_cbegin(row), weight,
                      this->getNumElements());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::addToSubset(
      const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        auto labelIterator =
          createBinarySparseForwardIterator(view.labelMatrix.indices_cbegin(row), view.labelMatrix.indices_cend(row));
        addInternally(this->in_begin(), this->ip_begin(), this->rn_begin(), this->rp_begin(), labelIterator,
                      view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend(),
                      view.coverageMatrix.values_cbegin(row), weight, this->getNumElements());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::addToSubset(
      const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
      const PartialIndexVector& indices, StatisticType weight) {
        auto majorityIterator =
          createBinarySparseForwardIterator(view.majorityLabelVector.cbegin(), view.majorityLabelVector.cend());
        auto coverageIterator = view.coverageMatrix.values_cbegin(row);
        auto labelIterator = view.labelMatrix.values_cbegin(row);
        auto indexIterator = indices.cbegin();
        uint32 numElements = indices.getNumElements();
        uint32 previousIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];
            uint32 coverage = coverageIterator[index];

            if (coverage == 0) {
                bool trueLabel = labelIterator[index];
                std::advance(majorityIterator, index - previousIndex);
                bool majorityLabel = *majorityIterator;
                StatisticType& element = getElement(trueLabel, majorityLabel, this->in_begin()[i], this->ip_begin()[i],
                                                    this->rn_begin()[i], this->rp_begin()[i]);
                element += weight;
                previousIndex = index;
            }
        }
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::addToSubset(
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
                StatisticType& element = getElement(trueLabel, majorityLabel, this->in_begin()[i], this->ip_begin()[i],
                                                    this->rn_begin()[i], this->rp_begin()[i]);
                element += weight;
                previousIndex = index;
            }
        }
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::difference(
      const DenseConfusionMatrixVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseConfusionMatrixVectorView<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        SequentialVectorMath::difference(this->in_begin(), first.in_cbegin(), second.in_cbegin(), numElements);
        SequentialVectorMath::difference(this->ip_begin(), first.ip_cbegin(), second.ip_cbegin(), numElements);
        SequentialVectorMath::difference(this->rn_begin(), first.rn_cbegin(), second.rn_cbegin(), numElements);
        SequentialVectorMath::difference(this->rp_begin(), first.rp_cbegin(), second.rp_cbegin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::difference(
      const DenseConfusionMatrixVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseConfusionMatrixVectorView<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        uint32 numElements = this->getNumElements();
        SequentialVectorMath::difference(this->in_begin(), first.in_cbegin(), second.in_cbegin(), indexIterator,
                                         numElements);
        SequentialVectorMath::difference(this->ip_begin(), first.ip_cbegin(), second.ip_cbegin(), indexIterator,
                                         numElements);
        SequentialVectorMath::difference(this->rn_begin(), first.rn_cbegin(), second.rn_cbegin(), indexIterator,
                                         numElements);
        SequentialVectorMath::difference(this->rp_begin(), first.rp_cbegin(), second.rp_cbegin(), indexIterator,
                                         numElements);
    }

    template class DenseConfusionMatrixVector<uint32, SequentialVectorMath>;
    template class DenseConfusionMatrixVector<float32, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseConfusionMatrixVector<uint32, SimdVectorMath>;
    template class DenseConfusionMatrixVector<float32, SimdVectorMath>;
#endif
}
