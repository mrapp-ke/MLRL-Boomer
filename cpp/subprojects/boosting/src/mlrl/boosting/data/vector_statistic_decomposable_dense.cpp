#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"

namespace boosting {

    DenseDecomposableStatisticVector::DenseDecomposableStatisticVector(uint32 numElements, bool init)
        : ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<Tuple<float64>>>>(
            AllocatedVector<Tuple<float64>>(numElements, init)) {}

    DenseDecomposableStatisticVector::DenseDecomposableStatisticVector(const DenseDecomposableStatisticVector& other)
        : DenseDecomposableStatisticVector(other.getNumElements()) {
        util::copyView(other.cbegin(), this->begin(), this->getNumElements());
    }

    void DenseDecomposableStatisticVector::add(const DenseDecomposableStatisticVector& vector) {
        util::addToView(this->begin(), vector.cbegin(), this->getNumElements());
    }

    void DenseDecomposableStatisticVector::add(const CContiguousView<Tuple<float64>>& view, uint32 row) {
        util::addToView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    void DenseDecomposableStatisticVector::add(const CContiguousView<Tuple<float64>>& view, uint32 row,
                                               float64 weight) {
        util::addToViewWeighted(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    void DenseDecomposableStatisticVector::remove(const CContiguousView<Tuple<float64>>& view, uint32 row) {
        util::removeFromView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    void DenseDecomposableStatisticVector::remove(const CContiguousView<Tuple<float64>>& view, uint32 row,
                                                  float64 weight) {
        util::removeFromView(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    void DenseDecomposableStatisticVector::addToSubset(const CContiguousView<Tuple<float64>>& view, uint32 row,
                                                       const CompleteIndexVector& indices) {
        util::addToView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    void DenseDecomposableStatisticVector::addToSubset(const CContiguousView<Tuple<float64>>& view, uint32 row,
                                                       const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        util::addToView(this->begin(), view.values_cbegin(row), indexIterator, this->getNumElements());
    }

    void DenseDecomposableStatisticVector::addToSubset(const CContiguousView<Tuple<float64>>& view, uint32 row,
                                                       const CompleteIndexVector& indices, float64 weight) {
        util::addToViewWeighted(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    void DenseDecomposableStatisticVector::addToSubset(const CContiguousView<Tuple<float64>>& view, uint32 row,
                                                       const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        util::addToViewWeighted(this->begin(), view.values_cbegin(row), indexIterator, this->getNumElements(), weight);
    }

    void DenseDecomposableStatisticVector::difference(const DenseDecomposableStatisticVector& first,
                                                      const CompleteIndexVector& firstIndices,
                                                      const DenseDecomposableStatisticVector& second) {
        util::setViewToDifference(this->begin(), first.cbegin(), second.cbegin(), this->getNumElements());
    }

    void DenseDecomposableStatisticVector::difference(const DenseDecomposableStatisticVector& first,
                                                      const PartialIndexVector& firstIndices,
                                                      const DenseDecomposableStatisticVector& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        util::setViewToDifference(this->begin(), first.cbegin(), second.cbegin(), indexIterator,
                                  this->getNumElements());
    }

}
