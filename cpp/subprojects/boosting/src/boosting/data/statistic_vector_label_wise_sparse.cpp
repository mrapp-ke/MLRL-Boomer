#include "boosting/data/statistic_vector_label_wise_sparse.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements)
        : SparseLabelWiseStatisticVector(numElements, false) {

    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements, bool init)
        : numElements_(numElements),
          statistics_((Triple<float64>*) (init ? calloc(numElements, sizeof(Triple<float64>))
                                               : malloc(numElements * sizeof(Triple<float64>)))),
          sumOfWeights_(0) {

    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(const SparseLabelWiseStatisticVector& vector)
        : SparseLabelWiseStatisticVector(vector.numElements_) {
        copyArray(vector.statistics_, statistics_, numElements_);
        sumOfWeights_ = vector.sumOfWeights_;
    }

    SparseLabelWiseStatisticVector::~SparseLabelWiseStatisticVector() {
        free(statistics_);
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cbegin() const {
        return statistics_;
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cend() const {
        return &statistics_[numElements_];
    }

    void SparseLabelWiseStatisticVector::clear() {
        sumOfWeights_ = 0;
        setArrayToZeros(statistics_, numElements_);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticVector& vector) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                             float64 weight) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
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
