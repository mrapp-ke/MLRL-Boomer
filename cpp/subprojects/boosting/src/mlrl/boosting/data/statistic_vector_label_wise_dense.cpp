#include "mlrl/boosting/data/statistic_vector_label_wise_dense.hpp"

#include "mlrl/boosting/util/view_functions.hpp"
#include "mlrl/common/util/memory.hpp"
#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements)
        : DenseLabelWiseStatisticVector(numElements, false) {}

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements, bool init)
        : numElements_(numElements), statistics_(allocateMemory<Tuple<float64>>(numElements, init)) {}

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& vector)
        : DenseLabelWiseStatisticVector(vector.numElements_) {
        copyArray(vector.statistics_, statistics_, numElements_);
    }

    DenseLabelWiseStatisticVector::~DenseLabelWiseStatisticVector() {
        freeMemory(statistics_);
    }

    DenseLabelWiseStatisticVector::iterator DenseLabelWiseStatisticVector::begin() {
        return statistics_;
    }

    DenseLabelWiseStatisticVector::iterator DenseLabelWiseStatisticVector::end() {
        return &statistics_[numElements_];
    }

    DenseLabelWiseStatisticVector::const_iterator DenseLabelWiseStatisticVector::cbegin() const {
        return statistics_;
    }

    DenseLabelWiseStatisticVector::const_iterator DenseLabelWiseStatisticVector::cend() const {
        return &statistics_[numElements_];
    }

    uint32 DenseLabelWiseStatisticVector::getNumElements() const {
        return numElements_;
    }

    void DenseLabelWiseStatisticVector::clear() {
        setArrayToZeros(statistics_, numElements_);
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticVector& vector) {
        addToArray(statistics_, vector.statistics_, numElements_);
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticConstView& view, uint32 row) {
        addToArray(statistics_, view.cbegin(row), numElements_);
    }

    void DenseLabelWiseStatisticVector::add(const DenseLabelWiseStatisticConstView& view, uint32 row, float64 weight) {
        addToArray(statistics_, view.cbegin(row), numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::remove(const DenseLabelWiseStatisticConstView& view, uint32 row) {
        removeFromArray(statistics_, view.cbegin(row), numElements_);
    }

    void DenseLabelWiseStatisticVector::remove(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                               float64 weight) {
        removeFromArray(statistics_, view.cbegin(row), numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const CompleteIndexVector& indices) {
        addToArray(statistics_, view.cbegin(row), numElements_);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToArray(statistics_, view.cbegin(row), indexIterator, numElements_);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const CompleteIndexVector& indices, float64 weight) {
        addToArray(statistics_, view.cbegin(row), numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                                                    const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToArray(statistics_, view.cbegin(row), indexIterator, numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::difference(const DenseLabelWiseStatisticVector& first,
                                                   const CompleteIndexVector& firstIndices,
                                                   const DenseLabelWiseStatisticVector& second) {
        setArrayToDifference(statistics_, first.cbegin(), second.cbegin(), numElements_);
    }

    void DenseLabelWiseStatisticVector::difference(const DenseLabelWiseStatisticVector& first,
                                                   const PartialIndexVector& firstIndices,
                                                   const DenseLabelWiseStatisticVector& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setArrayToDifference(statistics_, first.cbegin(), second.cbegin(), indexIterator, numElements_);
    }

}
