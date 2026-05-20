#include "mlrl/seco/data/vector_confusion_matrix_dense.hpp"

#include "mlrl/common/iterator/iterator_forward_sparse_binary.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/simd/vector_math.hpp"

namespace seco {

    template<typename T, typename Constant>
    static inline void addConstantToSubset(T* array, Constant constant, const uint32* indices, uint32 numIndices,
                                           const PartialIndexVector& indexVector) {
        auto indexVectorIterator = indexVector.cbegin();
        auto indexVectorEnd = indexVector.cend();
        uint32 n = 0;

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 index = indices[i];

            while (indexVectorIterator != indexVectorEnd && *indexVectorIterator < index) {
                indexVectorIterator++;
                n++;
            }

            if (indexVectorIterator != indexVectorEnd && *indexVectorIterator == index) {
                array[n] += constant;
            }
        }
    }

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

    template<typename StatisticType, typename VectorMath>
    DenseConfusionMatrixVector<StatisticType, VectorMath>::DenseConfusionMatrixVector(uint32 numElements, bool init)
        : ClearableViewDecorator<ViewDecorator<DenseConfusionMatrixVectorView<StatisticType>>>(
            DenseConfusionMatrixVectorView<StatisticType>(numElements, init)) {}

    template<typename StatisticType, typename VectorMath>
    DenseConfusionMatrixVector<StatisticType, VectorMath>::DenseConfusionMatrixVector(
      const DenseConfusionMatrixVector<StatisticType, VectorMath>& other)
        : DenseConfusionMatrixVector(other.getNumElements()) {
        uint32 numElements = this->getNumElements();
        VectorMath::copy(other.in_cbegin(), this->in_begin(), numElements);
        VectorMath::copy(other.ip_cbegin(), this->ip_begin(), numElements);
        VectorMath::copy(other.rn_cbegin(), this->rn_begin(), numElements);
        VectorMath::copy(other.rp_cbegin(), this->rp_begin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::add(
      const DenseConfusionMatrixVectorView<StatisticType>& other) {
        uint32 numElements = this->getNumElements();
        VectorMath::add(this->in_begin(), other.in_cbegin(), numElements);
        VectorMath::add(this->ip_begin(), other.ip_cbegin(), numElements);
        VectorMath::add(this->rn_begin(), other.rn_cbegin(), numElements);
        VectorMath::add(this->rp_begin(), other.rp_cbegin(), numElements);
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
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::add(const SparseDecomposableStatisticView& view,
                                                                    uint32 row, StatisticType weight) {
        VectorMath::addConstantToSubset(this->in_begin(), weight, view.in_indices_cbegin(row),
                                        view.in_indices_cend(row) - view.in_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->rn_begin(), weight, view.rn_indices_cbegin(row),
                                        view.rn_indices_cend(row) - view.rn_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->ip_begin(), weight, view.ip_indices_cbegin(row),
                                        view.ip_indices_cend(row) - view.ip_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->rp_begin(), weight, view.rp_indices_cbegin(row),
                                        view.rp_indices_cend(row) - view.rp_indices_cbegin(row));
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::remove(const SparseDecomposableStatisticView& view,
                                                                       uint32 row, StatisticType weight) {
        VectorMath::subtractConstantFromSubset(this->in_begin(), weight, view.in_indices_cbegin(row),
                                               view.in_indices_cend(row) - view.in_indices_cbegin(row));
        VectorMath::subtractConstantFromSubset(this->rn_begin(), weight, view.rn_indices_cbegin(row),
                                               view.rn_indices_cend(row) - view.rn_indices_cbegin(row));
        VectorMath::subtractConstantFromSubset(this->ip_begin(), weight, view.ip_indices_cbegin(row),
                                               view.ip_indices_cend(row) - view.ip_indices_cbegin(row));
        VectorMath::subtractConstantFromSubset(this->rp_begin(), weight, view.rp_indices_cbegin(row),
                                               view.rp_indices_cend(row) - view.rp_indices_cbegin(row));
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::addToSubset(const SparseDecomposableStatisticView& view,
                                                                            uint32 row,
                                                                            const CompleteIndexVector& indices,
                                                                            StatisticType weight) {
        VectorMath::addConstantToSubset(this->in_begin(), weight, view.in_indices_cbegin(row),
                                        view.in_indices_cend(row) - view.in_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->rn_begin(), weight, view.rn_indices_cbegin(row),
                                        view.rn_indices_cend(row) - view.rn_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->ip_begin(), weight, view.ip_indices_cbegin(row),
                                        view.ip_indices_cend(row) - view.ip_indices_cbegin(row));
        VectorMath::addConstantToSubset(this->rp_begin(), weight, view.rp_indices_cbegin(row),
                                        view.rp_indices_cend(row) - view.rp_indices_cbegin(row));
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::addToSubset(const SparseDecomposableStatisticView& view,
                                                                            uint32 row,
                                                                            const PartialIndexVector& indices,
                                                                            StatisticType weight) {
        addConstantToSubset(this->in_begin(), weight, view.in_indices_cbegin(row),
                            view.in_indices_cend(row) - view.in_indices_cbegin(row), indices);
        addConstantToSubset(this->rn_begin(), weight, view.rn_indices_cbegin(row),
                            view.rn_indices_cend(row) - view.rn_indices_cbegin(row), indices);
        addConstantToSubset(this->ip_begin(), weight, view.ip_indices_cbegin(row),
                            view.ip_indices_cend(row) - view.ip_indices_cbegin(row), indices);
        addConstantToSubset(this->rp_begin(), weight, view.rp_indices_cbegin(row),
                            view.rp_indices_cend(row) - view.rp_indices_cbegin(row), indices);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::difference(
      const DenseConfusionMatrixVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseConfusionMatrixVectorView<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        VectorMath::difference(this->in_begin(), first.in_cbegin(), second.in_cbegin(), numElements);
        VectorMath::difference(this->ip_begin(), first.ip_cbegin(), second.ip_cbegin(), numElements);
        VectorMath::difference(this->rn_begin(), first.rn_cbegin(), second.rn_cbegin(), numElements);
        VectorMath::difference(this->rp_begin(), first.rp_cbegin(), second.rp_cbegin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseConfusionMatrixVector<StatisticType, VectorMath>::difference(
      const DenseConfusionMatrixVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseConfusionMatrixVectorView<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        uint32 numElements = this->getNumElements();
        VectorMath::difference(this->in_begin(), first.in_cbegin(), second.in_cbegin(), indexIterator, numElements);
        VectorMath::difference(this->ip_begin(), first.ip_cbegin(), second.ip_cbegin(), indexIterator, numElements);
        VectorMath::difference(this->rn_begin(), first.rn_cbegin(), second.rn_cbegin(), indexIterator, numElements);
        VectorMath::difference(this->rp_begin(), first.rp_cbegin(), second.rp_cbegin(), indexIterator, numElements);
    }

    template class DenseConfusionMatrixVector<uint32, SequentialVectorMath>;
    template class DenseConfusionMatrixVector<float32, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseConfusionMatrixVector<uint32, SimdVectorMath>;
    template class DenseConfusionMatrixVector<float32, SimdVectorMath>;
#endif
}
