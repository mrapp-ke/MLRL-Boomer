#include "boosting/data/statistic_vector_label_wise_sparse.hpp"
#include "common/iterator/subset_forward_iterator.hpp"
#include "statistic_vector_label_wise_sparse_common.hpp"
#include <limits>


namespace boosting {

    static const uint32 LIMIT = std::numeric_limits<uint32>::max();

    template<typename Iterator>
    static inline uint32 fetchNextDifference(Iterator& firstIterator, Iterator firstEnd,
                                             SparseLabelWiseStatisticVector::const_iterator& secondIterator,
                                             SparseLabelWiseStatisticVector::const_iterator secondEnd,
                                             Triple<float64>& triple) {
        uint32 firstIndex = firstIterator == firstEnd ? LIMIT : (*firstIterator).index;
        uint32 secondIndex = secondIterator == secondEnd ? LIMIT : (*secondIterator).index;

        if (firstIndex < secondIndex) {
            triple = (*firstIterator).value;
            firstIterator++;
            return firstIndex;
        } else if (secondIndex < firstIndex) {
            triple = (*secondIterator).value;
            secondIterator++;
            return secondIndex;
        } else if (firstIndex < LIMIT) {
            triple = (*firstIterator).value - (*secondIterator).value;
            firstIterator++;
            secondIterator++;
            return firstIndex;
        }

        return LIMIT;
    }

    template<typename Iterator>
    static inline uint32 fetchNextNonZeroDifference(Iterator& firstIterator, Iterator firstEnd,
                                                    SparseLabelWiseStatisticVector::const_iterator& secondIterator,
                                                    SparseLabelWiseStatisticVector::const_iterator secondEnd,
                                                    Triple<float64>& triple) {
        uint32 index = fetchNextDifference(firstIterator, firstEnd, secondIterator, secondEnd, triple);

        while (triple.first == 0 && index < LIMIT) {
            index = fetchNextDifference(firstIterator, firstEnd, secondIterator, secondEnd, triple);
        }

        return index;
    }

    template<typename Iterator>
    static inline void differenceInternally(SparseListVector<Triple<float64>>& vector, Iterator firstBegin,
                                            Iterator firstEnd,
                                            SparseLabelWiseStatisticVector::const_iterator secondBegin,
                                            SparseLabelWiseStatisticVector::const_iterator secondEnd) {
        Triple<float64> triple;
        uint32 index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, triple);

        if (index < LIMIT) {
            SparseListVector<Triple<float64>>::iterator previous = vector.begin();
            SparseListVector<Triple<float64>>::iterator end = vector.end();
            SparseListVector<Triple<float64>>::iterator current = insertNext(vector, previous, end, index, triple);

            while (current != end) {
                index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, triple);

                if (index < LIMIT) {
                    insertNext(vector, previous, current, end, index, triple);
                } else {
                    vector.erase_after(previous, end);
                    return;
                }
            }

            while ((index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, triple)) < LIMIT) {
                previous = vector.emplace_after(previous, index, triple);
            }
        } else {
            vector.clear();
        }
    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements)
        : SparseLabelWiseStatisticVector(numElements, false) {

    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(uint32 numElements, bool init)
        : sumOfWeights_(0) {

    }

    SparseLabelWiseStatisticVector::SparseLabelWiseStatisticVector(const SparseLabelWiseStatisticVector& vector)
        : sumOfWeights_(vector.sumOfWeights_) {
        addToSparseLabelWiseStatisticVector<Triple<float64>, const_iterator>(
            vector_, vector.cbegin(), vector.cend(), 1);
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cbegin() const {
        return vector_.cbegin();
    }

    SparseLabelWiseStatisticVector::const_iterator SparseLabelWiseStatisticVector::cend() const {
        return vector_.cend();
    }

    void SparseLabelWiseStatisticVector::clear() {
        sumOfWeights_ = 0;
        vector_.clear();
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticVector& vector) {
        sumOfWeights_ += vector.sumOfWeights_;
        addToSparseLabelWiseStatisticVector<Triple<float64>, const_iterator>(
            vector_, vector.cbegin(), vector.cend(), 1);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row) {
        sumOfWeights_++;
        addToSparseLabelWiseStatisticVector<Tuple<float64>, SparseLabelWiseStatisticConstView::const_iterator>(
            vector_, view.row_cbegin(row), view.row_cend(row), 1);
    }

    void SparseLabelWiseStatisticVector::add(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                             float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector<Tuple<float64>, SparseLabelWiseStatisticConstView::const_iterator>(
                vector_, view.row_cbegin(row), view.row_cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addToSparseLabelWiseStatisticVector<Tuple<float64>, SparseLabelWiseStatisticConstView::const_iterator>(
                vector_, view.row_cbegin(row), view.row_cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            SparseLabelWiseStatisticConstView::const_iterator begin = view.row_cbegin(row);
            SparseLabelWiseStatisticConstView::const_iterator end = view.row_cend(row);
            PartialIndexVector::const_iterator indicesBegin = indices.cbegin();
            PartialIndexVector::const_iterator indicesEnd = indices.cend();
            auto subsetBegin =
                make_subset_forward_iterator<SparseLabelWiseStatisticConstView::const_iterator, Tuple<float64>,
                                             PartialIndexVector::const_iterator>(begin, end, indicesBegin, indicesEnd);
            auto subsetEnd =
                make_subset_forward_iterator<SparseLabelWiseStatisticConstView::const_iterator, Tuple<float64>,
                                             PartialIndexVector::const_iterator>(begin, end, indicesEnd, indicesEnd);
            addToSparseLabelWiseStatisticVector<Tuple<float64>,
                          SparseSubsetForwardIterator<SparseLabelWiseStatisticConstView::const_iterator, Tuple<float64>,
                                                      PartialIndexVector::const_iterator>>(vector_, subsetBegin,
                                                                                           subsetEnd, weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const CompleteIndexVector& indices, float64 weight) {
        float64 binWeight = view.getWeight(row) * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            addToSparseLabelWiseStatisticVector<Triple<float64>, SparseLabelWiseHistogramConstView::const_iterator>(
                vector_, view.row_cbegin(row), view.row_cend(row), weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                                                     const PartialIndexVector& indices, float64 weight) {
        float64 binWeight = view.getWeight(row) * weight;

        if (binWeight != 0) {
            sumOfWeights_ += binWeight;
            SparseLabelWiseHistogramConstView::const_iterator begin = view.row_cbegin(row);
            SparseLabelWiseHistogramConstView::const_iterator end = view.row_cend(row);
            PartialIndexVector::const_iterator indicesBegin = indices.cbegin();
            PartialIndexVector::const_iterator indicesEnd = indices.cend();
            auto subsetBegin =
                make_subset_forward_iterator<SparseLabelWiseHistogramConstView::const_iterator, Triple<float64>,
                                             PartialIndexVector::const_iterator>(begin, end, indicesBegin, indicesEnd);
            auto subsetEnd =
                make_subset_forward_iterator<SparseLabelWiseHistogramConstView::const_iterator, Triple<float64>,
                                             PartialIndexVector::const_iterator>(begin, end, indicesEnd, indicesEnd);
            addToSparseLabelWiseStatisticVector<Triple<float64>,
                          SparseSubsetForwardIterator<SparseLabelWiseHistogramConstView::const_iterator,
                                                      Triple<float64>, PartialIndexVector::const_iterator>>(vector_,
                                                                                                            subsetBegin,
                                                                                                            subsetEnd,
                                                                                                            weight);
        }
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        differenceInternally(
            vector_, first.cbegin(), first.cend(), second.cbegin(), second.cend());
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        SparseLabelWiseStatisticVector::const_iterator firstBegin = first.cbegin();
        SparseLabelWiseStatisticVector::const_iterator firstEnd = first.cend();
        PartialIndexVector::const_iterator indicesBegin = firstIndices.cbegin();
        PartialIndexVector::const_iterator indicesEnd = firstIndices.cend();
        auto subsetBegin =
            make_subset_forward_iterator<const_iterator, Triple<float64>, PartialIndexVector::const_iterator>(
                firstBegin, firstEnd, indicesBegin, indicesEnd);
        auto subsetEnd =
            make_subset_forward_iterator<const_iterator, Triple<float64>, PartialIndexVector::const_iterator>(
                firstBegin, firstEnd, indicesEnd, indicesEnd);
        differenceInternally(vector_, subsetBegin, subsetEnd, second.cbegin(), second.cend());
    }

    float64 SparseLabelWiseStatisticVector::getSumOfWeights() const {
        return sumOfWeights_;
    }

}
