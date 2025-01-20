#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseNonDecomposableStatisticVector<StatisticType>::DenseNonDecomposableStatisticVector(uint32 numGradients,
                                                                                            bool init)
        : ClearableViewDecorator<
            ViewDecorator<CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>>>(
            CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
              AllocatedVector<StatisticType>(numGradients, init),
              AllocatedVector<StatisticType>(util::triangularNumber(numGradients), init))) {}

    template<typename StatisticType>
    DenseNonDecomposableStatisticVector<StatisticType>::DenseNonDecomposableStatisticVector(
      const DenseNonDecomposableStatisticVector<StatisticType>& other)
        : DenseNonDecomposableStatisticVector<StatisticType>(other.getNumGradients()) {
        util::copyView(other.gradients_cbegin(), this->gradients_begin(), this->getNumGradients());
        util::copyView(other.hessians_cbegin(), this->hessians_begin(), this->getNumHessians());
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::gradients_begin() {
        return this->view.firstView.begin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::gradient_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::gradients_end() {
        return this->view.firstView.end();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::gradients_cbegin() const {
        return this->view.firstView.cbegin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::gradient_const_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::gradients_cend() const {
        return this->view.firstView.cend();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::hessians_begin() {
        return this->view.secondView.begin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::hessian_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::hessians_end() {
        return this->view.secondView.end();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::hessians_cbegin() const {
        return this->view.secondView.cbegin();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::hessian_const_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::hessians_cend() const {
        return this->view.secondView.cend();
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::hessians_diagonal_cbegin() const {
        return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin()), 0);
    }

    template<typename StatisticType>
    typename DenseNonDecomposableStatisticVector<StatisticType>::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticVector<StatisticType>::hessians_diagonal_cend() const {
        return hessian_diagonal_const_iterator(View<const StatisticType>(this->hessians_cbegin()),
                                               this->getNumHessians());
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticVector<StatisticType>::getNumGradients() const {
        return this->view.firstView.numElements;
    }

    template<typename StatisticType>
    uint32 DenseNonDecomposableStatisticVector<StatisticType>::getNumHessians() const {
        return this->view.secondView.numElements;
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::add(
      const DenseNonDecomposableStatisticVector<StatisticType>& view) {
        util::addToView(this->gradients_begin(), view.gradients_cbegin(), this->getNumGradients());
        util::addToView(this->hessians_begin(), view.hessians_cbegin(), this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::add(const DenseNonDecomposableStatisticView& view,
                                                                 uint32 row) {
        util::addToView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        util::addToView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::add(const DenseNonDecomposableStatisticView& view,
                                                                 uint32 row, StatisticType weight) {
        util::addToViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(), weight);
        util::addToViewWeighted(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::remove(const DenseNonDecomposableStatisticView& view,
                                                                    uint32 row) {
        util::removeFromView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        util::removeFromView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::remove(const DenseNonDecomposableStatisticView& view,
                                                                    uint32 row, StatisticType weight) {
        util::removeFromViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(),
                                     weight);
        util::removeFromViewWeighted(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(const DenseNonDecomposableStatisticView& view,
                                                                         uint32 row,
                                                                         const CompleteIndexVector& indices) {
        util::addToView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        util::addToView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(const DenseNonDecomposableStatisticView& view,
                                                                         uint32 row,
                                                                         const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        util::addToView(this->gradients_begin(), view.gradients_cbegin(row), indexIterator, this->getNumGradients());
        DenseNonDecomposableStatisticView::hessian_const_iterator hessiansBegin = view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            util::addToView(&this->hessians_begin()[util::triangularNumber(i)],
                            &hessiansBegin[util::triangularNumber(index)], indexIterator, i + 1);
        }
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(const DenseNonDecomposableStatisticView& view,
                                                                         uint32 row, const CompleteIndexVector& indices,
                                                                         StatisticType weight) {
        util::addToViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(), weight);
        util::addToViewWeighted(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::addToSubset(const DenseNonDecomposableStatisticView& view,
                                                                         uint32 row, const PartialIndexVector& indices,
                                                                         StatisticType weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        util::addToViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                this->getNumGradients(), weight);
        DenseNonDecomposableStatisticView::hessian_const_iterator hessiansBegin = view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            util::addToViewWeighted(&this->hessians_begin()[util::triangularNumber(i)],
                                    &hessiansBegin[util::triangularNumber(index)], indexIterator, i + 1, weight);
        }
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::difference(
      const DenseNonDecomposableStatisticVector<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVector<StatisticType>& second) {
        util::setViewToDifference(this->gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                                  this->getNumGradients());
        util::setViewToDifference(this->hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                                  this->getNumHessians());
    }

    template<typename StatisticType>
    void DenseNonDecomposableStatisticVector<StatisticType>::difference(
      const DenseNonDecomposableStatisticVector<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseNonDecomposableStatisticVector<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        util::setViewToDifference(this->gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                                  indexIterator, this->getNumGradients());
        DenseNonDecomposableStatisticVector<StatisticType>::hessian_const_iterator firstHessiansBegin =
          first.hessians_cbegin();
        DenseNonDecomposableStatisticVector<StatisticType>::hessian_const_iterator secondHessiansBegin =
          second.hessians_cbegin();

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 offset = util::triangularNumber(i);
            uint32 index = indexIterator[i];
            util::setViewToDifference(&this->hessians_begin()[offset],
                                      &firstHessiansBegin[util::triangularNumber(index)], &secondHessiansBegin[offset],
                                      indexIterator, i + 1);
        }
    }

    template class DenseNonDecomposableStatisticVector<float32>;
    template class DenseNonDecomposableStatisticVector<float64>;
}
