#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"

#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseDecomposableStatisticVector<StatisticType>::DenseDecomposableStatisticVector(uint32 numElements, bool init)
        : ClearableViewDecorator<
            ViewDecorator<CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>>>(
            CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>(
              AllocatedVector<StatisticType>(numElements, init), AllocatedVector<StatisticType>(numElements, init))) {}

    template<typename StatisticType>
    DenseDecomposableStatisticVector<StatisticType>::DenseDecomposableStatisticVector(
      const DenseDecomposableStatisticVector<StatisticType>& other)
        : DenseDecomposableStatisticVector<StatisticType>(other.getNumElements()) {
        util::copyView(other.gradients_cbegin(), this->gradients_begin(), this->getNumElements());
        util::copyView(other.hessians_cbegin(), this->hessians_begin(), this->getNumElements());
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::gradient_iterator
      DenseDecomposableStatisticVector<StatisticType>::gradients_begin() {
        return this->view.firstView.begin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::gradient_iterator
      DenseDecomposableStatisticVector<StatisticType>::gradients_end() {
        return this->view.firstView.end();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticVector<StatisticType>::gradients_cbegin() const {
        return this->view.firstView.cbegin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::gradient_const_iterator
      DenseDecomposableStatisticVector<StatisticType>::gradients_cend() const {
        return this->view.firstView.cend();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::hessian_iterator
      DenseDecomposableStatisticVector<StatisticType>::hessians_begin() {
        return this->view.secondView.begin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::hessian_iterator
      DenseDecomposableStatisticVector<StatisticType>::hessians_end() {
        return this->view.secondView.end();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticVector<StatisticType>::hessians_cbegin() const {
        return this->view.secondView.cbegin();
    }

    template<typename StatisticType>
    typename DenseDecomposableStatisticVector<StatisticType>::hessian_const_iterator
      DenseDecomposableStatisticVector<StatisticType>::hessians_cend() const {
        return this->view.secondView.cend();
    }

    template<typename StatisticType>
    uint32 DenseDecomposableStatisticVector<StatisticType>::getNumElements() const {
        return this->view.firstView.numElements;
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(
      const DenseDecomposableStatisticVector<StatisticType>& vector) {
        uint32 numElements = this->getNumElements();
        util::addToView(this->gradients_begin(), vector.gradients_cbegin(), numElements);
        util::addToView(this->hessians_begin(), vector.hessians_cbegin(), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(const DenseDecomposableStatisticView<StatisticType>& view,
                                                              uint32 row) {
        uint32 numElements = this->getNumElements();
        util::addToView(this->gradients_begin(), view.gradients_cbegin(row), numElements);
        util::addToView(this->hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(const DenseDecomposableStatisticView<StatisticType>& view,
                                                              uint32 row, StatisticType weight) {
        uint32 numElements = this->getNumElements();
        util::addToViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), numElements, weight);
        util::addToViewWeighted(this->hessians_begin(), view.hessians_cbegin(row), numElements, weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        uint32 numElements = this->getNumElements();
        util::removeFromView(this->gradients_begin(), view.gradients_cbegin(row), numElements);
        util::removeFromView(this->hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        uint32 numElements = this->getNumElements();
        util::removeFromViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), numElements, weight);
        util::removeFromViewWeighted(this->hessians_begin(), view.hessians_cbegin(row), numElements, weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        uint32 numElements = this->getNumElements();
        util::addToView(this->gradients_begin(), view.gradients_cbegin(row), numElements);
        util::addToView(this->hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        util::addToView(this->gradients_begin(), view.gradients_cbegin(row), indexIterator, numElements);
        util::addToView(this->hessians_begin(), view.hessians_cbegin(row), indexIterator, numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        uint32 numElements = this->getNumElements();
        util::addToViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), numElements, weight);
        util::addToViewWeighted(this->hessians_begin(), view.hessians_cbegin(row), numElements, weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        util::addToViewWeighted(this->gradients_begin(), view.gradients_cbegin(row), indexIterator, numElements,
                                weight);
        util::addToViewWeighted(this->hessians_begin(), view.hessians_cbegin(row), indexIterator, numElements, weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::difference(
      const DenseDecomposableStatisticVector<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseDecomposableStatisticVector<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        util::setViewToDifference(this->gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                                  numElements);
        util::setViewToDifference(this->hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                                  numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::difference(
      const DenseDecomposableStatisticVector<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseDecomposableStatisticVector<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        util::setViewToDifference(this->gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                                  indexIterator, numElements);
        util::setViewToDifference(this->hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                                  indexIterator, numElements);
    }

    template class DenseDecomposableStatisticVector<float32>;
    template class DenseDecomposableStatisticVector<float64>;
}
