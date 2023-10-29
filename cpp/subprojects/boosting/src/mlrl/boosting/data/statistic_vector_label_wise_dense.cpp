#include "mlrl/boosting/data/statistic_vector_label_wise_dense.hpp"

#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements, bool init)
        : WritableVectorDecorator<AllocatedView<Vector<Tuple<float64>>>>(
          AllocatedView<Vector<Tuple<float64>>>(numElements, init)) {}

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& other)
        : DenseLabelWiseStatisticVector(other.view_.numElements) {
        copyView(other.view_.array, this->view_.array, this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::clear() {
        setViewToZeros(this->view_.array, this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticVector& vector) {
        addToView(this->view_.array, vector.view_.array, this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticConstView& view, uint32 row) {
        addToView(this->view_.array, view.cbegin(row), this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticConstView& view, uint32 row, float64 weight) {
        addToView(this->view_.array, view.cbegin(row), this->view_.numElements, weight);
    }

    void DenseLabelWiseStatisticVector::remove(const DenseLabelWiseStatisticConstView& view, uint32 row) {
        removeFromView(this->view_.array, view.cbegin(row), this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::remove(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                               float64 weight) {
        removeFromView(this->view_.array, view.cbegin(row), this->view_.numElements, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const CompleteIndexVector& indices) {
        addToView(this->view_.array, view.cbegin(row), this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->view_.array, view.cbegin(row), indexIterator, this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const CompleteIndexVector& indices, float64 weight) {
        addToView(this->view_.array, view.cbegin(row), this->view_.numElements, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->view_.array, view.cbegin(row), indexIterator, this->view_.numElements, weight);
    }

    void DenseLabelWiseStatisticVector::difference(const DenseLabelWiseStatisticVector& first,
                                                   const CompleteIndexVector& firstIndices,
                                                   const DenseLabelWiseStatisticVector& second) {
        setViewToDifference(this->view_.array, first.cbegin(), second.cbegin(), this->view_.numElements);
    }

    void DenseLabelWiseStatisticVector::difference(const DenseLabelWiseStatisticVector& first,
                                                   const PartialIndexVector& firstIndices,
                                                   const DenseLabelWiseStatisticVector& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setViewToDifference(this->view_.array, first.cbegin(), second.cbegin(), indexIterator, this->view_.numElements);
    }

}
