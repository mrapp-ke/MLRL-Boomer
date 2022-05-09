/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    /**
     * Adds the statistics that are stored in a single row of a `SparseLabelWiseStatisticConstView` to a sparse vector.
     * The statistics are multiplied by a specific weight.
     *
     * @param statistics    A pointer to an array the statistics should be added to
     * @param begin         A `SparseLabelWiseStatisticsConstView::const_iterator` to the beginning of the statistics to
     *                      be added
     * @param begin         A `SparseLabelWiseStatisticsConstView::const_iterator` to the end of the statistics to be
     *                      added
     * @param weight        The weight, the statistics should be multiplied by
     */
    static inline void addToSparseLabelWiseStatisticVector(Triple<float64>* statistics,
                                                           SparseLabelWiseStatisticConstView::const_iterator begin,
                                                           SparseLabelWiseStatisticConstView::const_iterator end,
                                                           float64 weight) {
        uint32 numElements = end - begin;

        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<Tuple<float64>>& entry = begin[i];
            const Tuple<float64>& tuple = entry.value;
            Triple<float64>& triple = statistics[entry.index];
            triple.first += (tuple.first * weight);
            triple.second += (tuple.second * weight);
            triple.third += weight;
        }
    }

}
