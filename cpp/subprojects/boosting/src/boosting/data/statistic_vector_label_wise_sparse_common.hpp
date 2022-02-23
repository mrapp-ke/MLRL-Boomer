/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/triple.hpp"
#include "common/data/tuple.hpp"
#include "common/data/vector_sparse_list.hpp"


namespace boosting {

    static inline Triple<float64> createTriple(const Triple<float64>& triple, float64 weight) {
        return Triple<float64>(triple.first * weight, triple.second * weight, triple.third * weight);
    }

    static inline Triple<float64> createTriple(const Tuple<float64>& tuple, float64 weight) {
        return Triple<float64>(tuple.first * weight, tuple.second * weight, weight);
    }

    template<typename ValueType, typename Iterator>
    static inline void addToSparseLabelWiseStatisticVector(SparseListVector<Triple<float64>>& vector, Iterator iterator,
                                                           Iterator end, float64 weight) {
        if (iterator != end) {
            SparseListVector<Triple<float64>>::iterator previous = vector.begin();
            SparseListVector<Triple<float64>>::iterator last = vector.end();

            const IndexedValue<ValueType>& firstEntry = *iterator;
            SparseListVector<Triple<float64>>::iterator current = addFirst<Triple<float64>>(
                vector, previous, last, firstEntry.index, createTriple(firstEntry.value, weight));
            iterator++;

            while (current != last) {
                if (iterator != end) {
                    const IndexedValue<ValueType>& entry = *iterator;
                    add<Triple<float64>>(vector, previous, current, last, entry.index,
                                         createTriple(entry.value, weight));
                    iterator++;
                } else {
                    return;
                }
            }

            for (; iterator != end; iterator++) {
                const IndexedValue<ValueType>& entry = *iterator;
                previous = vector.emplace_after(previous, entry.index, createTriple(entry.value, weight));
            }
        }
    }

}
