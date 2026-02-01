#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/util/math.hpp"
#include "mlrl/common/util/array_operations.hpp"
#include "mlrl/common/util/xsimd.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseNonDecomposableStatisticVectorView<StatisticType>::DenseNonDecomposableStatisticVectorView(uint32 numGradients,
                                                                                                    bool init)
        : CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
            AllocatedVector<StatisticType>(numGradients, init),
            AllocatedVector<StatisticType>(util::triangularNumber(numGradients), init)) {}

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

    template<typename StatisticType, typename ArrayOperations>
    DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::DenseNonDecomposableStatisticVector(
      uint32 numGradients, bool init)
        : ClearableViewDecorator<ViewDecorator<DenseNonDecomposableStatisticVectorView<StatisticType>>>(
            DenseNonDecomposableStatisticVectorView<StatisticType>(numGradients, init)) {}

    template<typename StatisticType, typename ArrayOperations>
    DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::DenseNonDecomposableStatisticVector(
      const DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>& other)
        : DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>(other.getNumGradients()) {
        ArrayOperations::copy(other.view.gradients_cbegin(), this->view.gradients_begin(), this->getNumGradients());
        ArrayOperations::copy(other.view.hessians_cbegin(), this->view.hessians_begin(), this->getNumHessians());
    }

    template<typename StatisticType, typename ArrayOperations>
    uint32 DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::getNumGradients() const {
        return this->view.getNumGradients();
    }

    template<typename StatisticType, typename ArrayOperations>
    uint32 DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::getNumHessians() const {
        return this->view.getNumHessians();
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::add(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& vector) {
        ArrayOperations::add(this->view.gradients_begin(), vector.gradients_cbegin(), this->getNumGradients());
        ArrayOperations::add(this->view.hessians_begin(), vector.hessians_cbegin(), this->getNumHessians());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        ArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        ArrayOperations::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        ArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(),
                                     weight);
        ArrayOperations::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(),
                                     weight);
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        ArrayOperations::subtract(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        ArrayOperations::subtract(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        ArrayOperations::subtractWeighted(this->view.gradients_begin(), view.gradients_cbegin(row),
                                          this->getNumGradients(), weight);
        ArrayOperations::subtractWeighted(this->view.hessians_begin(), view.hessians_cbegin(row),
                                          this->getNumHessians(), weight);
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        ArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        ArrayOperations::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        ArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                             this->getNumGradients());
        typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator hessiansBegin =
          view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            ArrayOperations::add(&this->view.hessians_begin()[util::triangularNumber(i)],
                                 &hessiansBegin[util::triangularNumber(index)], indexIterator, i + 1);
        }
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        ArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(),
                                     weight);
        ArrayOperations::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(),
                                     weight);
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        ArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                     this->getNumGradients(), weight);
        typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator hessiansBegin =
          view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            ArrayOperations::addWeighted(&this->view.hessians_begin()[util::triangularNumber(i)],
                                         &hessiansBegin[util::triangularNumber(index)], indexIterator, i + 1, weight);
        }
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::difference(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVectorView<StatisticType>& second) {
        ArrayOperations::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                                    this->getNumGradients());
        ArrayOperations::difference(this->view.hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                                    this->getNumHessians());
    }

    template<typename StatisticType, typename ArrayOperations>
    void DenseNonDecomposableStatisticVector<StatisticType, ArrayOperations>::difference(
      const DenseNonDecomposableStatisticVectorView<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVectorView<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        ArrayOperations::difference(this->view.gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                                    indexIterator, this->getNumGradients());
        typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator firstHessiansBegin =
          first.hessians_cbegin();
        typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator secondHessiansBegin =
          second.hessians_cbegin();

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 offset = util::triangularNumber(i);
            uint32 index = indexIterator[i];
            ArrayOperations::difference(&this->view.hessians_begin()[offset],
                                        &firstHessiansBegin[util::triangularNumber(index)],
                                        &secondHessiansBegin[offset], indexIterator, i + 1);
        }
    }

    template class DenseNonDecomposableStatisticVector<float32, SequentialArrayOperations>;
    template class DenseNonDecomposableStatisticVector<float64, SequentialArrayOperations>;

#if SIMD_SUPPORT_ENABLED
    template class DenseNonDecomposableStatisticVector<float32, SimdArrayOperations>;
    template class DenseNonDecomposableStatisticVector<float64, SimdArrayOperations>;
#endif
}
