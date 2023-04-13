/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/transformation_binary.hpp"

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms regression scores into binary predictions
     * via element-wise comparison to a predefined threshold.
     */
    class LabelWiseBinaryTransformation final : public IBinaryTransformation {
        private:

            const float64 threshold_;

        public:

            /**
             * @param threshold The threshold to be used for discretization
             */
            LabelWiseBinaryTransformation(float64 threshold);

            void apply(VectorConstView<float64>::const_iterator scoresBegin,
                       VectorConstView<float64>::const_iterator scoresEnd, VectorView<uint8>::iterator predictionBegin,
                       VectorView<uint8>::iterator predictionEnd) const override;

            void apply(VectorConstView<float64>::const_iterator scoresBegin,
                       VectorConstView<float64>::const_iterator scoresEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
