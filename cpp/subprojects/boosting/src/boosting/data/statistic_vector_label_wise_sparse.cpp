#include "boosting/data/statistic_vector_label_wise_sparse.hpp"
#include "common/iterator/subset_forward_iterator.hpp"
#include <limits>


namespace boosting {

    static const uint32 LIMIT = std::numeric_limits<uint32>::max();

    template<typename ValueType, typename Iterator>
    static inline void addInternally(SparseListVector<AggregatedStatistics>& vector, Iterator iterator, Iterator end,
                                     float64 weight) {
        if (iterator != end) {
            SparseListVector<AggregatedStatistics>::iterator previous = vector.begin();
            SparseListVector<AggregatedStatistics>::iterator last = vector.end();

            const IndexedValue<ValueType>& firstEntry = *iterator;
            SparseListVector<AggregatedStatistics>::iterator current = addFirst<AggregatedStatistics>(
                vector, previous, last, firstEntry.index, AggregatedStatistics(firstEntry.value, weight));
            iterator++;

            while (current != last) {
                if (iterator != end) {
                    const IndexedValue<ValueType>& entry = *iterator;
                    add<AggregatedStatistics>(vector, previous, current, last, entry.index,
                                              AggregatedStatistics(entry.value, weight));
                    iterator++;
                } else {
                    return;
                }
            }

            for (; iterator != end; iterator++) {
                const IndexedValue<ValueType>& entry = *iterator;
                previous = vector.emplace_after(previous, entry.index, AggregatedStatistics(entry.value, weight));
            }
        }
    }

    template<typename Iterator>
    static inline uint32 fetchNextDifference(Iterator& firstIterator, Iterator firstEnd,
                                             SparseLabelWiseStatisticVector::const_iterator& secondIterator,
                                             SparseLabelWiseStatisticVector::const_iterator secondEnd,
                                             AggregatedStatistics& statistics) {
        uint32 firstIndex = firstIterator == firstEnd ? LIMIT : (*firstIterator).index;
        uint32 secondIndex = secondIterator == secondEnd ? LIMIT : (*secondIterator).index;

        if (firstIndex < secondIndex) {
            statistics = (*firstIterator).value;
            firstIterator++;
            return firstIndex;
        } else if (secondIndex < firstIndex) {
            statistics = (*secondIterator).value;
            secondIterator++;
            return secondIndex;
        } else if (firstIndex < LIMIT) {
            statistics = (*firstIterator).value - (*secondIterator).value;
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
                                                    AggregatedStatistics& statistics) {
        uint32 index = fetchNextDifference(firstIterator, firstEnd, secondIterator, secondEnd, statistics);

        while (statistics.sumOfGradients == 0 && index < LIMIT) {
            index = fetchNextDifference(firstIterator, firstEnd, secondIterator, secondEnd, statistics);
        }

        return index;
    }

    template<typename Iterator>
    static inline void differenceInternally(SparseListVector<AggregatedStatistics>& vector, Iterator firstBegin,
                                            Iterator firstEnd,
                                            SparseLabelWiseStatisticVector::const_iterator secondBegin,
                                            SparseLabelWiseStatisticVector::const_iterator secondEnd) {
        AggregatedStatistics statistics;
        uint32 index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, statistics);

        if (index < LIMIT) {
            SparseListVector<AggregatedStatistics>::iterator previous = vector.begin();
            SparseListVector<AggregatedStatistics>::iterator end = vector.end();
            SparseListVector<AggregatedStatistics>::iterator current = insertNext(vector, previous, end, index,
                                                                                  statistics);

            while (current != end) {
                index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, statistics);

                if (index < LIMIT) {
                    insertNext(vector, previous, current, end, index, statistics);
                } else {
                    vector.erase_after(previous, end);
                    return;
                }
            }

            while ((index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, statistics))
                   < LIMIT) {
                previous = vector.emplace_after(previous, index, statistics);
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
        addInternally<AggregatedStatistics, const_iterator>(vector_, vector.cbegin(), vector.cend(), 1);
    }

    void SparseLabelWiseStatisticVector::add(SparseLabelWiseStatisticConstView::const_iterator begin,
                                             SparseLabelWiseStatisticConstView::const_iterator end) {
        sumOfWeights_++;
        addInternally<Tuple<float64>, SparseLabelWiseStatisticConstView::const_iterator>(vector_, begin, end, 1);
    }

    void SparseLabelWiseStatisticVector::add(SparseLabelWiseStatisticConstView::const_iterator begin,
                                             SparseLabelWiseStatisticConstView::const_iterator end, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addInternally<Tuple<float64>, SparseLabelWiseStatisticConstView::const_iterator>(
                vector_, begin, end, weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(SparseLabelWiseStatisticConstView::const_iterator begin,
                                                     SparseLabelWiseStatisticConstView::const_iterator end,
                                                     const CompleteIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            addInternally<Tuple<float64>, SparseLabelWiseStatisticConstView::const_iterator>(
                vector_, begin, end, weight);
        }
    }

    void SparseLabelWiseStatisticVector::addToSubset(SparseLabelWiseStatisticConstView::const_iterator begin,
                                                     SparseLabelWiseStatisticConstView::const_iterator end,
                                                     const PartialIndexVector& indices, float64 weight) {
        if (weight != 0) {
            sumOfWeights_ += weight;
            PartialIndexVector::const_iterator indicesBegin = indices.cbegin();
            PartialIndexVector::const_iterator indicesEnd = indices.cend();
            auto subsetBegin =
                make_subset_forward_iterator<SparseLabelWiseStatisticConstView::const_iterator, Tuple<float64>,
                                             PartialIndexVector::const_iterator>(begin, end, indicesBegin, indicesEnd);
            auto subsetEnd =
                make_subset_forward_iterator<SparseLabelWiseStatisticConstView::const_iterator, Tuple<float64>,
                                             PartialIndexVector::const_iterator>(begin, end, indicesEnd, indicesEnd);
            addInternally<Tuple<float64>,
                          SparseSubsetForwardIterator<SparseLabelWiseStatisticConstView::const_iterator, Tuple<float64>,
                                                      PartialIndexVector::const_iterator>>(vector_, subsetBegin,
                                                                                           subsetEnd, weight);
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
            make_subset_forward_iterator<const_iterator, AggregatedStatistics, PartialIndexVector::const_iterator>(
                firstBegin, firstEnd, indicesBegin, indicesEnd);
        auto subsetEnd =
            make_subset_forward_iterator<const_iterator, AggregatedStatistics, PartialIndexVector::const_iterator>(
                firstBegin, firstEnd, indicesEnd, indicesEnd);
        differenceInternally(vector_, subsetBegin, subsetEnd, second.cbegin(), second.cend());
    }

    float64 SparseLabelWiseStatisticVector::getSumOfWeights() const {
        return sumOfWeights_;
    }

}
