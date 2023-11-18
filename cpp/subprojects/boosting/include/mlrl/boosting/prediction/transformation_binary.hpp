/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_lil_binary.hpp"
#include "mlrl/common/data/view_vector.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform regression scores into binary predictions.
     */
    class IBinaryTransformation {
        public:

            virtual ~IBinaryTransformation() {};

            /**
             * Transforms regression scores into binary predictions.
             *
             * @param scoresBegin       An iterator to the beginning of the regression scores
             * @param scoresEnd         An iterator to the end of the regression scores
             * @param predictionBegin   An iterator to the beginning of the binary predictions
             * @param predictionEnd     An iterator to the end of the binary predictions
             */
            virtual void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                               View<uint8>::iterator predictionBegin, View<uint8>::iterator predictionEnd) const = 0;

            /**
             * Transforms regression scores into sparse binary predictions.
             *
             * @param scoresBegin   An iterator to the beginning of the regression scores
             * @param scoresEnd     An iterator to the end of the regression scores
             * @param predictionRow An object of type `BinaryLilMatrix::row` that should be used to store the binary
             *                      predictions
             */
            virtual void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                               BinaryLilMatrix::row predictionRow) const = 0;
    };

}
