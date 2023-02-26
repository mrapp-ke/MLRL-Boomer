/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/transformation_binary.hpp"

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms real-valued predictions into binary
     * predictions via element-wise comparison to a predefined threshold.
     */
    class LabelWiseBinaryTransformation final : public IBinaryTransformation {
        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold to be used for discretization
             */
            LabelWiseBinaryTransformation(float64 threshold);

            void apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                       CContiguousConstView<float64>::value_const_iterator realEnd,
                       CContiguousView<uint8>::value_iterator predictionBegin,
                       CContiguousView<uint8>::value_iterator predictionEnd) const override;

            void apply(CContiguousConstView<float64>::value_const_iterator realBegin,
                       CContiguousConstView<float64>::value_const_iterator realEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
