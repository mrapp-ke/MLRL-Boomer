#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/util/math.hpp"
#include "mlrl/common/util/array_operations.hpp"

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

    template<typename StatisticType>
    DenseNonDecomposableStatisticVector<StatisticType>::DenseNonDecomposableStatisticVector(uint32 numGradients,
                                                                                            bool init)
        : ClearableViewDecorator<ViewDecorator<DenseNonDecomposableStatisticVectorView<StatisticType>>>(
            DenseNonDecomposableStatisticVectorView<StatisticType>(numGradients, init)) {}

    template<typename StatisticType>
    DenseNonDecomposableStatisticVector<StatisticType>::DenseNonDecomposableStatisticVector(
      const DenseNonDecomposableStatisticVector<StatisticType>& other)
        : DenseNonDecomposableStatisticVector<StatisticType>(other.getNumGradients()) {
        SequentialArrayOperations::copy(other.view.gradients_cbegin(), this->view.gradients_begin(),
                                        this->getNumGradients());
        SequentialArrayOperations::copy(other.view.hessians_cbegin(), this->view.hessians_begin(),
                                        this->getNumHessians());
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticVector<StatisticType>::getNumGradients() const {
        return this->view.getNumGradients();
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticVector<StatisticType>::getNumHessians() const {
        return this->view.getNumHessians();
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::add(
      const DenseNonDecomposableStatisticVector<StatisticType>& vector) {
        SequentialArrayOperations::add(this->view.gradients_begin(), vector.view.gradients_cbegin(),
                                       this->getNumGradients());
        SequentialArrayOperations::add(this->view.hessians_begin(), vector.view.hessians_cbegin(),
                                       this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        SequentialArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row),
                                       this->getNumGradients());
        SequentialArrayOperations::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::add(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        SequentialArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row),
                                               this->getNumGradients(), weight);
        SequentialArrayOperations::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row),
                                               this->getNumHessians(), weight);
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row) {
        SequentialArrayOperations::subtract(this->view.gradients_begin(), view.gradients_cbegin(row),
                                            this->getNumGradients());
        SequentialArrayOperations::subtract(this->view.hessians_begin(), view.hessians_cbegin(row),
                                            this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::remove(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        SequentialArrayOperations::subtractWeighted(this->view.gradients_begin(), view.gradients_cbegin(row),
                                                    this->getNumGradients(), weight);
        SequentialArrayOperations::subtractWeighted(this->view.hessians_begin(), view.hessians_cbegin(row),
                                                    this->getNumHessians(), weight);
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        SequentialArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row),
                                       this->getNumGradients());
        SequentialArrayOperations::add(this->view.hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        SequentialArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                       this->getNumGradients());
        typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator hessiansBegin =
          view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            SequentialArrayOperations::add(&this->view.hessians_begin()[util::triangularNumber(i)],
                                           &hessiansBegin[util::triangularNumber(index)], indexIterator, i + 1);
        }
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        SequentialArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row),
                                               this->getNumGradients(), weight);
        SequentialArrayOperations::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row),
                                               this->getNumHessians(), weight);
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        SequentialArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                               this->getNumGradients(), weight);
        typename DenseNonDecomposableStatisticView<StatisticType>::hessian_const_iterator hessiansBegin =
          view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            SequentialArrayOperations::addWeighted(&this->view.hessians_begin()[util::triangularNumber(i)],
                                                   &hessiansBegin[util::triangularNumber(index)], indexIterator, i + 1,
                                                   weight);
        }
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::difference(
      const DenseNonDecomposableStatisticVector<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVector<StatisticType>& second) {
        SequentialArrayOperations::difference(this->view.gradients_begin(), first.view.gradients_cbegin(),
                                              second.view.gradients_cbegin(), this->getNumGradients());
        SequentialArrayOperations::difference(this->view.hessians_begin(), first.view.hessians_cbegin(),
                                              second.view.hessians_cbegin(), this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::difference(
      const DenseNonDecomposableStatisticVector<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVector<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        SequentialArrayOperations::difference(this->view.gradients_begin(), first.view.gradients_cbegin(),
                                              second.view.gradients_cbegin(), indexIterator, this->getNumGradients());
        typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator firstHessiansBegin =
          first.view.hessians_cbegin();
        typename DenseNonDecomposableStatisticVectorView<StatisticType>::hessian_const_iterator secondHessiansBegin =
          second.view.hessians_cbegin();

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 offset = util::triangularNumber(i);
            uint32 index = indexIterator[i];
            SequentialArrayOperations::difference(&this->view.hessians_begin()[offset],
                                                  &firstHessiansBegin[util::triangularNumber(index)],
                                                  &secondHessiansBegin[offset], indexIterator, i + 1);
        }
    }

    template class DenseNonDecomposableStatisticVector<float32>;
    template class DenseNonDecomposableStatisticVector<float64>;
}
