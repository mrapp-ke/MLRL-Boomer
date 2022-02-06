#include "boosting/data/statistic_vector_label_wise_sparse.hpp"


namespace boosting {

    AggregatedStatistics::AggregatedStatistics() {

    }

    AggregatedStatistics::AggregatedStatistics(float64 g, float64 h, uint32 n)
        : sumOfGradients(g), sumOfHessians(h), numAggregatedStatistics(n) {

    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements)
        : SparseLabelWiseStatisticVector(numElements, false) {

    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements, bool init)
        : numAggregatedStatistics_(0) {

    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cbegin() const {
        return vector_.cbegin();
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cend() const {
        return vector_.cend();
    }

    void SparseLabelWiseStatisticVector::clear() {
        vector_.clear();
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticVector& vector) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::add(SparseLabelWiseStatisticConstView::const_iterator begin,
                                             SparseLabelWiseStatisticConstView::const_iterator end) {
        // TODO Implement
        numAggregatedStatistics_++;
    }

    void SparseLabelWiseStatisticVector::add(SparseLabelWiseStatisticConstView::const_iterator begin,
                                             SparseLabelWiseStatisticConstView::const_iterator end, float64 weight) {
        if (weight != 0) {
            // TODO Implement
            numAggregatedStatistics_++;
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(SparseLabelWiseStatisticConstView::const_iterator begin,
                                                     SparseLabelWiseStatisticConstView::const_iterator end,
                                                     const CompleteIndexVector& indices, float64 weight) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::addToSubset(SparseLabelWiseStatisticConstView::const_iterator begin,
                                                     SparseLabelWiseStatisticConstView::const_iterator end,
                                                     const PartialIndexVector& indices, float64 weight) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        // TODO Implement
    }

}
