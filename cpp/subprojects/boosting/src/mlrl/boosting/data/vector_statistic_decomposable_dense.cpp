#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"

namespace boosting {

    template<typename StatisticType>
    DenseDecomposableStatisticVector<StatisticType>::DenseDecomposableStatisticVector(uint32 numElements, bool init)
        : ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<Statistic<StatisticType>>>>(
            AllocatedVector<Statistic<StatisticType>>(numElements, init)) {}

    template<typename StatisticType>
    DenseDecomposableStatisticVector<StatisticType>::DenseDecomposableStatisticVector(
      const DenseDecomposableStatisticVector<StatisticType>& other)
        : DenseDecomposableStatisticVector<StatisticType>(other.getNumElements()) {
        util::copyView(other.cbegin(), this->begin(), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(
      const DenseDecomposableStatisticVector<StatisticType>& vector) {
        util::addToView(this->begin(), vector.cbegin(), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(const DenseDecomposableStatisticView<StatisticType>& view,
                                                              uint32 row) {
        // TODO util::addToView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::add(const DenseDecomposableStatisticView<StatisticType>& view,
                                                              uint32 row, StatisticType weight) {
        // TODO util::addToViewWeighted(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row) {
        // TODO util::removeFromView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::remove(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight) {
        // TODO util::removeFromViewWeighted(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices) {
        // TODO util::addToView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices) {
        // TODO util::addToView(this->begin(), view.values_cbegin(row), indices.cbegin(), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const CompleteIndexVector& indices,
      StatisticType weight) {
        // TODO util::addToViewWeighted(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::addToSubset(
      const DenseDecomposableStatisticView<StatisticType>& view, uint32 row, const PartialIndexVector& indices,
      StatisticType weight) {
        // TODO util::addToViewWeighted(this->begin(), view.values_cbegin(row), indices.cbegin(),
        // this->getNumElements(), weight);
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::difference(
      const DenseDecomposableStatisticVector<StatisticType>& first, const CompleteIndexVector& firstIndices,
      const DenseDecomposableStatisticVector<StatisticType>& second) {
        // TODO util::setViewToDifference(this->begin(), first.cbegin(), second.cbegin(), this->getNumElements());
    }

    template<typename StatisticType>
    void DenseDecomposableStatisticVector<StatisticType>::difference(
      const DenseDecomposableStatisticVector<StatisticType>& first, const PartialIndexVector& firstIndices,
      const DenseDecomposableStatisticVector<StatisticType>& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        util::setViewToDifference(this->begin(), first.cbegin(), second.cbegin(), indexIterator,
                                  this->getNumElements());
    }

    template class DenseDecomposableStatisticVector<float32>;
    template class DenseDecomposableStatisticVector<float64>;
}
