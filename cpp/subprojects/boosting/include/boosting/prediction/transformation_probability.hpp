/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform aggregated scores into probability estimates.
     */
    class IProbabilityTransformation {
        public:

            virtual ~IProbabilityTransformation() {};

            /**
             * Transforms aggregated scores into probability estimates.
             *
             * @param scoresBegin           An iterator of type `CContiguousConstView::value_const_iterator` to the
             *                              beginning of the aggregated scores
             * @param scoresEnd             An iterator of type `CContiguousConstView::value_const_iterator` to the end
             *                              of the aggregated scores
             * @param probabilitiesBegin    An iterator of type `CContiguousView::value_iterator` to the beginning of
             *                              the probabilities
             * @param probabilitiesEnd      An iterator of type `CContiguousView::value_iterator` to the end of the
             *                              probabilities
             */
            virtual void apply(CContiguousConstView<float64>::value_const_iterator scoresBegin,
                               CContiguousConstView<float64>::value_const_iterator scoresEnd,
                               CContiguousView<float64>::value_iterator probabilitiesBegin,
                               CContiguousView<float64>::value_iterator probabilitiesEnd) const = 0;
    };

}
