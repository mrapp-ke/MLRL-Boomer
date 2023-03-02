/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_lil_binary.hpp"
#include "common/data/view_c_contiguous.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform real-valued predictions into binary predictions.
     */
    class IBinaryTransformation {
        public:

            virtual ~IBinaryTransformation() {};

            /**
             * Transforms real-valued predictions into binary predictions.
             *
             * @param realBegin         An iterator of type `CContiguousConstView::value_const_iterator` to the
             *                          beginning of the real-valued predictions
             * @param realEnd           An iterator of type `CContiguousConstView::value_const_iterator` to the end of
             *                          the real-valued predictions
             * @param predictionBegin   An iterator of type `CContiguousView::value_iterator` to the beginning of the
             *                          binary predictions
             * @param predictionEnd     An iterator of type `CContiguousView::value_iterator` to the end of the binary
             *                          predictions
             */
            virtual void apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                               CContiguousConstView<float64>::value_const_iterator realEnd,
                               CContiguousView<uint8>::value_iterator predictionBegin,
                               CContiguousView<uint8>::value_iterator predictionEnd) const = 0;

            /**
             * Transforms real-valued predictions into sparse binary predictions.
             *
             * @param realBegin     An iterator of type `CContiguousConstView::value_const_iterator` to the beginning of
             *                      the real-valued predictions
             * @param realEnd       An iterator of type `CContiguousConstView::value_const_iterator` to the end of the
             *                      real-valued predictions
             * @param predictionRow An object of type `BinaryLilMatrix::row` that should be used to store the binary
             *                      predictions
             */
            virtual void apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                               CContiguousConstView<float64>::value_const_iterator realEnd,
                               BinaryLilMatrix::row predictionRow) const = 0;
    };

}
