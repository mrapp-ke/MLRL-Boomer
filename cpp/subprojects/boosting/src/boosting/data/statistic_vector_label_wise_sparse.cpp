#include "boosting/data/statistic_vector_label_wise_sparse.hpp"


namespace boosting {

    static inline void add(SparseListVector<AggregatedStatistics>& vector,
                           SparseListVector<AggregatedStatistics>::iterator& previous,
                           SparseListVector<AggregatedStatistics>::iterator& current,
                           SparseListVector<AggregatedStatistics>::iterator end, uint32 index,
                           const AggregatedStatistics& value) {
        uint32 currentIndex = advance<AggregatedStatistics>(previous, current, end, index);

        if (index == currentIndex) {
            (*current).value += value;
        } else if (index > currentIndex) {
            current = vector.emplace_after(current, index, value);
        } else {
            current = vector.emplace_after(previous, index, value);
        }

        previous = current;
        current++;
    }

    static inline SparseListVector<AggregatedStatistics>::iterator addFirst(
            SparseListVector<AggregatedStatistics>& vector, SparseListVector<AggregatedStatistics>::iterator& begin,
            SparseListVector<AggregatedStatistics>::iterator end, uint32 index, const AggregatedStatistics& value) {
        if (begin == end) {
            vector.emplace_front(index, value);
            begin = vector.begin();
        } else {
            IndexedValue<AggregatedStatistics>& entry = *begin;
            uint32 firstIndex = entry.index;

            if (index == firstIndex) {
                entry.value += value;
            } else if (index < firstIndex) {
                vector.emplace_front(index, value);
                begin = vector.begin();
            } else {
                SparseListVector<AggregatedStatistics>::iterator current = begin;
                current++;
                add(vector, begin, current, end, index, value);
                return current;
            }
        }

        SparseListVector<AggregatedStatistics>::iterator current = begin;
        current++;
        return current;
    }

    template<typename ValueType, typename Iterator>
    static inline void addInternally(SparseListVector<AggregatedStatistics>& vector, Iterator begin, Iterator end,
                                     float64 weight) {
        if (begin != end) {
            SparseListVector<AggregatedStatistics>::iterator previous = vector.begin();
            SparseListVector<AggregatedStatistics>::iterator last = vector.end();

            const IndexedValue<ValueType>& firstEntry = *begin;
            SparseListVector<AggregatedStatistics>::iterator current =
                addFirst(vector, previous, last, firstEntry.index, AggregatedStatistics(firstEntry.value, weight));
            begin++;

            for (; begin != end && current != last; begin++) {
                const IndexedValue<ValueType>& entry = *begin;
                add(vector, previous, current, last, entry.index, AggregatedStatistics(entry.value, weight));
            }

            for (; begin != end; begin++) {
                const IndexedValue<ValueType>& entry = *begin;
                previous = vector.emplace_after(previous, entry.index, AggregatedStatistics(entry.value, weight));
            }
        }
    }

    static inline uint32 fetchNextDifference(SparseLabelWiseStatisticVector::const_iterator& firstIterator,
                                             SparseLabelWiseStatisticVector::const_iterator firstEnd,
                                             SparseLabelWiseStatisticVector::const_iterator& secondIterator,
                                             SparseLabelWiseStatisticVector::const_iterator secondEnd,
                                             AggregatedStatistics& statistics, uint32 limit) {
        uint32 firstIndex = firstIterator == firstEnd ? limit : (*firstIterator).index;
        uint32 secondIndex = secondIterator == secondEnd ? limit : (*secondIterator).index;

        if (firstIndex < secondIndex) {
            statistics = (*firstIterator).value;
            firstIterator++;
            return firstIndex;
        } else if (secondIndex < firstIndex) {
            statistics = (*secondIterator).value;
            secondIterator++;
            return secondIndex;
        } else if (firstIndex < limit) {
            statistics = (*firstIterator).value - (*secondIterator).value;
            firstIterator++;
            secondIterator++;
            return firstIndex;
        }

        return limit;
    }

    static inline uint32 fetchNextNonZeroDifference(SparseLabelWiseStatisticVector::const_iterator& firstIterator,
                                                    SparseLabelWiseStatisticVector::const_iterator firstEnd,
                                                    SparseLabelWiseStatisticVector::const_iterator& secondIterator,
                                                    SparseLabelWiseStatisticVector::const_iterator secondEnd,
                                                    AggregatedStatistics& statistics, uint32 limit) {
        uint32 index = fetchNextDifference(firstIterator, firstEnd, secondIterator, secondEnd, statistics, limit);

        while (statistics.sumOfGradients == 0 && index < limit) {
            index = fetchNextDifference(firstIterator, firstEnd, secondIterator, secondEnd, statistics, limit);
        }

        return index;
    }

    static inline void differenceInternally(SparseListVector<AggregatedStatistics>& vector,
                                            SparseLabelWiseStatisticVector::const_iterator firstBegin,
                                            SparseLabelWiseStatisticVector::const_iterator firstEnd,
                                            SparseLabelWiseStatisticVector::const_iterator secondBegin,
                                            SparseLabelWiseStatisticVector::const_iterator secondEnd, uint32 limit) {
        AggregatedStatistics statistics;
        uint32 index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, statistics, limit);

        if (index < limit) {
            insertNext(vector, index, statistics);
            SparseListVector<AggregatedStatistics>::iterator previous = vector.begin();

            while ((index = fetchNextNonZeroDifference(firstBegin, firstEnd, secondBegin, secondEnd, statistics, limit))
                   < limit) {
                previous = insertNext(vector, index, statistics, previous);
            }

            vector.erase_after(previous, vector.end());
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
        // TODO Implement
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const CompleteIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        sumOfWeights_ = first.sumOfWeights_ - second.sumOfWeights_;
        differenceInternally(
            vector_, first.cbegin(), first.cend(), second.cbegin(), second.cend(), firstIndices.getNumElements());
    }

    void SparseLabelWiseStatisticVector::difference(const SparseLabelWiseStatisticVector& first,
                                                    const PartialIndexVector& firstIndices,
                                                    const SparseLabelWiseStatisticVector& second) {
        // TODO Implement
    }

    float64 SparseLabelWiseStatisticVector::getSumOfWeights() const {
        return sumOfWeights_;
    }

}
