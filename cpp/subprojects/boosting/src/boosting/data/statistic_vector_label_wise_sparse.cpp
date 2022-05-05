#include "boosting/data/statistic_vector_label_wise_sparse.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>
#include <iostream>  // TODO Remove


namespace boosting {

    SparseLabelWiseStatisticVector::ConstIterator::ConstIterator(const Triple<float64>* iterator, float64 sumOfWeights)
        : iterator_(iterator), sumOfWeights_(sumOfWeights) {

    }

    SparseLabelWiseStatisticVector::ConstIterator::value_type SparseLabelWiseStatisticVector::ConstIterator::operator[](
            uint32 index) const {
        const Triple<float64>& triple = iterator_[index];
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseLabelWiseStatisticVector::ConstIterator::value_type SparseLabelWiseStatisticVector::ConstIterator::operator*() const {
        const Triple<float64>& triple = *iterator_;
        float64 gradient = triple.first;
        float64 hessian = triple.second + (sumOfWeights_ - triple.third);
        return Tuple<float64>(gradient, hessian);
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator++() {
        ++iterator_;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator++(int n) {
        iterator_++;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator--() {
        --iterator_;
        return *this;
    }

    SparseLabelWiseStatisticVector::ConstIterator& SparseLabelWiseStatisticVector::ConstIterator::operator--(int n) {
        iterator_--;
        return *this;
    }

    bool SparseLabelWiseStatisticVector::ConstIterator::operator!=(const ConstIterator& rhs) const {
        return iterator_ != rhs.iterator_;
    }

    bool SparseLabelWiseStatisticVector::ConstIterator::operator==(const ConstIterator& rhs) const {
        return iterator_ == rhs.iterator_;
    }

    SparseLabelWiseStatisticVector::ConstIterator::difference_type SparseLabelWiseStatisticVector::ConstIterator::operator-(
            const ConstIterator& rhs) const {
        return iterator_ - rhs.iterator_;
    }

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
        return ConstIterator(statistics_, sumOfWeights_);
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cend() const {
        return ConstIterator(&statistics_[numElements_], sumOfWeights_);
    }

    uint32 SparseLabelWiseStatisticVector::getNumElements() const {
        return numElements_;
    }

    void SparseLabelWiseStatisticVector::clear() {
        sumOfWeights_ = 0;
        setArrayToZeros(statistics_, numElements_);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticVector& vector) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::add(SparseLabelWiseStatisticVector)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::add(SparseLabelWiseStatisticConstView)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                             float64 weight) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::add(SparseLabelWiseStatisticConstView, weight)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::addToSubset(CompleteIndexVector)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::addToSubset(PartialIndexVector)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::addToSubset(SparseLabelWiseHistogramConstView, CompleteIndexVector)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::addToSubset(SparseLabelWiseHistogramConstView, PartialIndexVector)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::difference(CompleteIndexVector)\n";
        std::exit(-1);
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        // TODO Implement
        std::cout << "SparseLabelWiseStatisticVector::difference(PartialIndexVector)\n";
        std::exit(-1);
    }

}
