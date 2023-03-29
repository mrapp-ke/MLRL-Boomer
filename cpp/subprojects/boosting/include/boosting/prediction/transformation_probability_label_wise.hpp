/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function_label_wise.hpp"
#include "boosting/prediction/transformation_probability.hpp"

namespace boosting {

    /**
     * An implementation of the class `IProbabilityTransformation` that transforms aggregated scores into probability
     * estimates via element-wise application of a `ILabelWiseProbabilityFunction`.
     */
    class LabelWiseProbabilityTransformation final : public IProbabilityTransformation {
        private:

            const std::unique_ptr<ILabelWiseProbabilityFunction> labelWiseProbabilityFunctionPtr_;

        public:

            /**
             * @param labelWiseProbabilityFunctionPtr An unique pointer to an object of type
             *                                        `ILabelWiseProbabilityFunction` that should be used to transform
             *                                        regression scores that are predicted for individual labels into
             *                                        probabilities
             */
            LabelWiseProbabilityTransformation(
              std::unique_ptr<ILabelWiseProbabilityFunction> labelWiseProbabilityFunctionPtr);

            void apply(VectorConstView<float64>::const_iterator scoresBegin,
                       VectorConstView<float64>::const_iterator scoresEnd,
                       VectorView<float64>::iterator probabilitiesBegin,
                       VectorView<float64>::iterator probabilitiesEnd) const override;
    };

}
