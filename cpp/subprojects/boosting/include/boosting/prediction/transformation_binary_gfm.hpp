/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function_marginal.hpp"
#include "boosting/prediction/transformation_binary.hpp"
#include "common/prediction/label_vector_set.hpp"

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms real-valued predictions into binary
     * predictions according to the general F-measure maximizer (GFM).
     */
    class GfmBinaryTransformation final : public IBinaryTransformation {
        private:

            const LabelVectorSet& labelVectorSet_;

            const uint32 maxLabelCardinality_;

            const std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr_;

        public:

            /**
             * @param labelVectorSet                    A reference to an object of type `LabelVectorSet` that stores
             *                                          all known label vectors
             * @param marginalProbabilityFunctionPtr    An unique pointer to an object of type
             *                                          `ILabelWise ProbabilityFunction` that should be used to
             *                                          transform regression scores that are predicted for individual
             *                                          labels into probabilities
             */
            GfmBinaryTransformation(const LabelVectorSet& labelVectorSet,
                                    std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr);

            void apply(VectorConstView<float64>::const_iterator realBegin,
                       VectorConstView<float64>::const_iterator realEnd, VectorView<uint8>::iterator predictionBegin,
                       VectorView<uint8>::iterator predictionEnd) const override;

            void apply(VectorConstView<float64>::const_iterator realBegin,
                       VectorConstView<float64>::const_iterator realEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
