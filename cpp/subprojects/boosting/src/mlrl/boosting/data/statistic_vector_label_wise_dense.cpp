#include "mlrl/boosting/data/statistic_vector_label_wise_dense.hpp"

namespace boosting {

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements, bool init)
        : ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<Tuple<float64>>>>(
          AllocatedVector<Tuple<float64>>(numElements, init)) {}

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& other)
        : DenseLabelWiseStatisticVector(other.getNumElements()) {
        copyView(other.cbegin(), this->begin(), this->getNumElements());
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticVector& vector) {
        addToView(this->begin(), vector.cbegin(), this->getNumElements());
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticView& view, uint32 row) {
        addToView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticView& view, uint32 row, float64 weight) {
        addToView(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    void DenseLabelWiseStatisticVector::remove(const DenseLabelWiseStatisticView& view, uint32 row) {
        removeFromView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    void DenseLabelWiseStatisticVector::remove(const DenseLabelWiseStatisticView& view, uint32 row, float64 weight) {
        removeFromView(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticView& view, uint32 row,
                                                    const CompleteIndexVector& indices) {
        addToView(this->begin(), view.values_cbegin(row), this->getNumElements());
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticView& view, uint32 row,
                                                    const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->begin(), view.values_cbegin(row), indexIterator, this->getNumElements());
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticView& view, uint32 row,
                                                    const CompleteIndexVector& indices, float64 weight) {
        addToView(this->begin(), view.values_cbegin(row), this->getNumElements(), weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticView& view, uint32 row,
                                                    const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->begin(), view.values_cbegin(row), indexIterator, this->getNumElements(), weight);
    }

    void DenseLabelWiseStatisticVector::difference(const DenseLabelWiseStatisticVector& first,
                                                   const CompleteIndexVector& firstIndices,
                                                   const DenseLabelWiseStatisticVector& second) {
        setViewToDifference(this->begin(), first.cbegin(), second.cbegin(), this->getNumElements());
    }

    void DenseLabelWiseStatisticVector::difference(const DenseLabelWiseStatisticVector& first,
                                                   const PartialIndexVector& firstIndices,
                                                   const DenseLabelWiseStatisticVector& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setViewToDifference(this->begin(), first.cbegin(), second.cbegin(), indexIterator, this->getNumElements());
    }

}
