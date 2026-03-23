#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/math/scalar_math.hpp"
#include "mlrl/common/math/vector_math.hpp"
#include "mlrl/common/util/xsimd.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseNonDecomposableStatisticVectorView<StatisticType>::DenseNonDecomposableStatisticVectorView(uint32 numGradients,
                                                                                                    bool init)
        : CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
            AllocatedVector<StatisticType>(numGradients, init),
            AllocatedVector<StatisticType>(math::triangularNumber(numGradients), init)) {}

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::gradients_begin() {
        return this->firstView.begin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::gradients_end() {
        return this->firstView.end();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::gradients_cbegin() const {
        return this->firstView.cbegin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::gradients_cend() const {
        return this->firstView.cend();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::hessians_begin() {
        return this->secondView.begin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::hessians_end() {
        return this->secondView.end();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::hessians_cbegin() const {
        return this->secondView.cbegin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::hessians_cend() const {
        return this->secondView.cend();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::hessians_diagonal_cbegin() const {
        return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin()), 0);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticVectorView<StatisticType>::hessians_diagonal_cend() const {
        return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin()),
                                               this->getNumHessians());
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticVectorView<StatisticType>::getNumGradients() const {
        return this->firstView.numElements;
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticVectorView<StatisticType>::getNumHessians() const {
        return this->secondView.numElements;
    }

    template class DenseNonDecomposableStatisticVectorView<float32>;
    template class DenseNonDecomposableStatisticVectorView<float64>;

    template<typename StatisticType, typename VectorMath>
    DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::DenseNonDecomposableStatisticVector(
      uint32 numGradients, bool init)
        : ClearableViewDecorator<ViewDecorator<DenseNonDecomposableStatisticVectorView<StatisticType>>>(
            DenseNonDecomposableStatisticVectorView<StatisticType>(numGradients, init)) {}

    template<typename StatisticType, typename VectorMath>
    DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::DenseNonDecomposableStatisticVector(
      const DenseNonDecomposableStatisticVector<StatisticType, VectorMath>& other)
        : DenseNonDecomposableStatisticVector<StatisticType, VectorMath>(other.getNumGradients()) {
        VectorMath::copy(other.view.gradients_cbegin(), this->view.gradients_begin(), this->getNumGradients());
        VectorMath::copy(other.view.hessians_cbegin(), this->view.hessians_begin(), this->getNumHessians());
    }

    template<typename StatisticType, typename VectorMath>
    uint32 DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::getNumGradients() const {
        return this->view.getNumGradients();
    }

    template<typename StatisticType, typename VectorMath>
    uint32 DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::getNumHessians() const {
        return this->view.getNumHessians();
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& vector) {
        VectorMath::add(this->view.gradients_begin(), vector.gradients_cbegin(), this->getNumGradients());
        VectorMath::add(this->view.hessians_begin(), vector.hessians_cbegin(), this->getNumHessians());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        VectorMath::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(),
                                weight);
        VectorMath::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        VectorMath::subtract(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        VectorMath::subtract(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        VectorMath::subtractWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(),
                                     weight);
        VectorMath::subtractWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(),
                                     weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        VectorMath::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        VectorMath::add(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                        this->getNumGradients());
        typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator hessiansBegin =
          view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            VectorMath::add(&this->view.hessians_begin()[math::triangularNumber(i)],
                            &hessiansBegin[math::triangularNumber(index)], indexIterator, i + 1);
        }
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(),
                                weight);
        VectorMath::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        VectorMath::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                this->getNumGradients(), weight);
        typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator hessiansBegin =
          view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            VectorMath::addWeighted(&this->view.hessians_begin()[math::triangularNumber(i)],
                                    &hessiansBegin[math::triangularNumber(index)], indexIterator, i + 1, weight);
        }
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVectorView<StatisticType>& second) {
        VectorMath::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                               this->getNumGradients());
        VectorMath::difference(this->view.hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                               this->getNumHessians());
    }

    template<typename StatisticType, typename VectorMath>
    void DenseNonDecomposableStatisticVector<StatisticType, VectorMath>::difference(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVectorView<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        VectorMath::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                               indexIterator, this->getNumGradients());
        typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator firstHessiansBegin =
          first.hessians_cbegin();
        typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator secondHessiansBegin =
          second.hessians_cbegin();

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 offset = math::triangularNumber(i);
            uint32 index = indexIterator[i];
            VectorMath::difference(&this->view.hessians_begin()[offset],
                                   &firstHessiansBegin[math::triangularNumber(index)], &secondHessiansBegin[offset],
                                   indexIterator, i + 1);
        }
    }

    template class DenseNonDecomposableStatisticVector<float32, SequentialVectorMath>;
    template class DenseNonDecomposableStatisticVector<float64, SequentialVectorMath>;

#if SIMD_SUPPORT_ENABLED
    template class DenseNonDecomposableStatisticVector<float32, SimdVectorMath>;
    template class DenseNonDecomposableStatisticVector<float64, SimdVectorMath>;
#endif
}
