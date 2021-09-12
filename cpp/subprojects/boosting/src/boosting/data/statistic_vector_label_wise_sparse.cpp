#include "boosting/data/statistic_vector_label_wise_sparse.hpp"


namespace boosting {

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements) {

    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements, bool init) {

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

    void SparseLabelWiseStatisticVector::add(const_iterator begin, const_iterator end) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::add(const_iterator begin, const_iterator end, float64 weight) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::addToSubset(const_iterator begin, const_iterator end,
                                                     const CompleteIndexVector& indices, float64 weight) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::addToSubset(const_iterator begin, const_iterator end,
                                                     const PartialIndexVector& indices, float64 weight) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                    const CompleteIndexVector& firstIndices, const_iterator secondBegin,
                                                    const_iterator secondEnd) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::difference(const_iterator firstBegin, const_iterator firstEnd,
                                                    const PartialIndexVector& firstIndices, const_iterator secondBegin,
                                                    const_iterator secondEnd) {
        // TODO Implement
    }

}
