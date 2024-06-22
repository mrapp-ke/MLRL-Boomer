/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/probability_function_joint.hpp"
#include "mlrl/boosting/prediction/transformation_binary.hpp"
#include "mlrl/common/prediction/label_vector_set.hpp"

#include <memory>

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms scores into binary predictions according
     * to the general F-measure maximizer (GFM).
     */
    class GfmBinaryTransformation final : public IBinaryTransformation {
        private:

            const LabelVectorSet& labelVectorSet_;

            const uint32 maxLabelCardinality_;

            const std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr_;

        public:

            /**
             * @param labelVectorSet                A reference to an object of type `LabelVectorSet` that stores all
             *                                      known label vectors
             * @param jointProbabilityFunctionPtr   An unique pointer to an object of type `JointProbabilityFunction`
             *                                      that should be used to transform scores that are predicted for an
             *                                      example into a joint probability
             */
            GfmBinaryTransformation(const LabelVectorSet& labelVectorSet,
                                    std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr);

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       View<uint8>::iterator predictionBegin, View<uint8>::iterator predictionEnd) const override;

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
