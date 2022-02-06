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

    AggregatedStatistics::AggregatedStatistics(const AggregatedStatistics& aggregatedStatistics, float64 weight)
        : sumOfGradients(aggregatedStatistics.sumOfGradients * weight),
          sumOfHessians(aggregatedStatistics.sumOfHessians * weight), sumOfWeights(weight) {

    }

    AggregatedStatistics::AggregatedStatistics(const Tuple<float64>& tuple, float64 weight)
        : sumOfGradients(tuple.first * weight), sumOfHessians(tuple.second * weight), sumOfWeights(weight) {

    }

    AggregatedStatistics& AggregatedStatistics::operator+=(const AggregatedStatistics& rhs) {
        sumOfGradients += rhs.sumOfGradients;
        sumOfHessians += rhs.sumOfHessians;
        sumOfWeights += rhs.sumOfWeights;
        return *this;
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

    float64 SparseLabelWiseStatisticVector::getSumOfWeights() const {
        return sumOfWeights_;
    }

}
