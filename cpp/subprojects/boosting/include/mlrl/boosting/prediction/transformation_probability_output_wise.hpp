/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/probability_function_marginal.hpp"
#include "mlrl/boosting/prediction/transformation_probability.hpp"

#include <memory>

namespace boosting {

    /**
     * An implementation of the class `IProbabilityTransformation` that transforms scores that have been aggregated for
     * individual labels into probability estimates via element-wise application of an `IMarginalProbabilityFunction`.
     */
    class OutputWiseProbabilityTransformation final : public IProbabilityTransformation {
        private:

            const std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr_;

        public:

            /**
             * @param marginalProbabilityFunctionPtr An unique pointer to an object of type
             *                                       `IMarginalProbabilityFunction` that should be used to transform
             *                                       scores that are predicted for individual labels into probabilities
             */
            OutputWiseProbabilityTransformation(
              std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr);

            void apply(View<float64>::const_iterator scoresBegin, View<float64>::const_iterator scoresEnd,
                       View<float64>::iterator probabilitiesBegin,
                       View<float64>::iterator probabilitiesEnd) const override;
    };

}
