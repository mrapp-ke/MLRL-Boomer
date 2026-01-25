#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"

#include "mlrl/common/util/array_operations.hpp"

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

    template<typename StatisticType>
    DenseDecomposableStatisticVector<StatisticType>::DenseDecomposableStatisticVector(uint32 numElements, bool init)
        : ClearableViewDecorator<ViewDecorator<DenseDecomposableStatisticVectorView<StatisticType>>>(
            DenseDecomposableStatisticVectorView<StatisticType>(numElements, init)) {}

    template<typename StatisticType>
    DenseDecomposableStatisticVector<StatisticType>::DenseDecomposableStatisticVector(
      const DenseDecomposableStatisticVector<StatisticType>& other)
        : DenseDecomposableStatisticVector<StatisticType>(other.getNumElements()) {
        SequentialArrayOperations::copy(other.view.gradients_cbegin(), this->view.gradients_begin(),
                                        this->getNumElements());
        SequentialArrayOperations::copy(other.view.hessians_cbegin(), this->view.hessians_begin(),
                                        this->getNumElements());
    }

    template<typename StatisticType>
    uint32 DenseDecomposableStatisticVector<StatisticType>::getNumElements() const {
        return this->view.getNumElements();
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(
      const DenseDecomposableStatisticVector<StatisticType>& vector) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::add(this->view.gradients_begin(), vector.view.gradients_cbegin(), numElements);
        SequentialArrayOperations::add(this->view.hessians_begin(), vector.view.hessians_cbegin(), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(const DenseDecomposableStatisticView<StatisticType>& view,
                                                              uint32 row) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row), numElements);
        SequentialArrayOperations::add(this->view.hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(const DenseDecomposableStatisticView<StatisticType>& view,
                                                              uint32 row, StatisticType weight) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), numElements,
                                               weight);
        SequentialArrayOperations::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), numElements,
                                               weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::subtract(this->view.gradients_begin(), view.gradients_cbegin(row), numElements);
        SequentialArrayOperations::subtract(this->view.hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::subtractWeighted(this->view.gradients_begin(), view.gradients_cbegin(row),
                                                    numElements, weight);
        SequentialArrayOperations::subtractWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), numElements,
                                                    weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row), numElements);
        SequentialArrayOperations::add(this->view.hessians_begin(), view.hessians_cbegin(row), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        SequentialArrayOperations::add(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                       numElements);
        SequentialArrayOperations::add(this->view.hessians_begin(), view.hessians_cbegin(row), indexIterator,
                                       numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), numElements,
                                               weight);
        SequentialArrayOperations::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), numElements,
                                               weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        SequentialArrayOperations::addWeighted(this->view.gradients_begin(), view.gradients_cbegin(row), indexIterator,
                                               numElements, weight);
        SequentialArrayOperations::addWeighted(this->view.hessians_begin(), view.hessians_cbegin(row), indexIterator,
                                               numElements, weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::difference(
      const DenseDecomposableStatisticVector<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseDecomposableStatisticVector<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        SequentialArrayOperations::difference(this->view.gradients_begin(), first.view.gradients_cbegin(),
                                              second.view.gradients_cbegin(), numElements);
        SequentialArrayOperations::difference(this->view.hessians_begin(), first.view.hessians_cbegin(),
                                              second.view.hessians_cbegin(), numElements);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::difference(
      const DenseDecomposableStatisticVector<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseDecomposableStatisticVector<StatisticType>& second) {
        uint32 numElements = this->getNumElements();
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        SequentialArrayOperations::difference(this->view.gradients_begin(), first.view.gradients_cbegin(),
                                              second.view.gradients_cbegin(), indexIterator, numElements);
        SequentialArrayOperations::difference(this->view.hessians_begin(), first.view.hessians_cbegin(),
                                              second.view.hessians_cbegin(), indexIterator, numElements);
    }

    template class DenseDecomposableStatisticVector<float32>;
    template class DenseDecomposableStatisticVector<float64>;
}
