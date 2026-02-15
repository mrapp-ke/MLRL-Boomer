#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"

#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/util/xsimd.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseDecomposableStatisticVectorView<StatisticType>::DenseDecomposableStatisticVectorView(uint32 numElements,
                                                                                              bool init)
        : CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
            AllocatedVector<StatisticType>(numElements, init), AllocatedVector<StatisticType>(numElements, init)) {}

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::gradient_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::gradients_begin() {
        return this->firstView.begin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::gradient_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::gradients_end() {
        return this->firstView.end();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::gradients_cbegin() const {
        return this->firstView.cbegin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::gradients_cend() const {
        return this->firstView.cend();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::hessian_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::hessians_begin() {
        return this->secondView.begin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::hessian_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::hessians_end() {
        return this->secondView.end();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::hessians_cbegin() const {
        return this->secondView.cbegin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticVectorView<StatisticType>::hessians_cend() const {
        return this->secondView.cend();
    }

    template<typename StatisticType>
    uint32 DenseDecomposableStatisticVectorView<StatisticType>::getNumElements() const {
        return this->firstView.numElements;
    }

    template class DenseDecomposableStatisticVectorView<float32>;
    template class DenseDecomposableStatisticVectorView<float64>;

    template<typename StatisticType, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, VectorMath>::DenseDecomposableStatisticVector(uint32 numElements,
                                                                                                  bool init)
        : ClearableViewDecorator<ViewDecorator<DenseDecomposableStatisticVectorView<StatisticType>>>(
            DenseDecomposableStatisticVectorView<StatisticType>(numElements, init)) {}

    template<typename StatisticType, typename VectorMath>
    DenseDecomposableStatisticVector<StatisticType, VectorMath>::DenseDecomposableStatisticVector(
      const DenseDecomposableStatisticVector<StatisticType, VectorMath>& other)
        : DenseDecomposableStatisticVector<StatisticType, VectorMath>(other.getNumElements()) {
        VectorMath::copy(other.view.gradients_cbegin(), this->view.gradients_begin(), this->getNumElements());
        VectorMath::copy(other.view.hessians_cbegin(), this->view.hessians_begin(), this->getNumElements());
    }

    template<typename StatisticType, typename VectorMath>
    uint32 DenseDecomposableStatisticVector<StatisticType, VectorMath>::getNumElements() const {
        return this->view.getNumElements();
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseDecomposableStatisticVectorView<StatisticType>& vector) {
        uint32 numElements = this->getNumElements();
        VectorMath::add(this->view.gradients_begin(), vector.gradients_cbegin(), numElements);
        VectorMath::add(this->view.hessians_begin(), vector.hessians_cbegin(), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        uint32 numElements = this->getNumElements();
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), numElements);
        VectorMath::add(this->view.hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        uint32 numElements = this->getNumElements();
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), numElements, weight);
        VectorMath::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), numElements, weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        uint32 numElements = this->getNumElements();
        VectorMath::subtract(this->view.gradients_begin(), view.gradients_cbegin(row), numElements);
        VectorMath::subtract(this->view.hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        uint32 numElements = this->getNumElements();
        VectorMath::subtractWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), numElements, weight);
        VectorMath::subtractWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), numElements, weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        uint32 numElements = this->getNumElements();
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), numElements);
        VectorMath::add(this->view.hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator, numElements);
        VectorMath::add(this->view.hessians_begin(), view.hessians_cbegin(row), indexIterator, numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        uint32 numElements = this->getNumElements();
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), numElements, weight);
        VectorMath::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), numElements, weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator, numElements,
                                weight);
        VectorMath::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), indexIterator, numElements,
                                weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        VectorMath::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                               numElements);
        VectorMath::difference(this->view.hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                               numElements);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseDecomposableStatisticVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseDecomposableStatisticVectorView<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        VectorMath::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                               indexIterator, numElements);
        VectorMath::difference(this->view.hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                               indexIterator, numElements);
    }

    template class DenseDecomposableStatisticVector<float32, SequentialVectorMath>;
    template class DenseDecomposableStatisticVector<float64, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseDecomposableStatisticVector<float32, SimdVectorMath>;
    template class DenseDecomposableStatisticVector<float64, SimdVectorMath>;
#endif
}
