/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function.hpp"
#include "boosting/prediction/transformation_probability.hpp"

namespace boosting {

    /**
     * An implementation of the class `IProbabilityTransformation` that transforms aggregated scores into probability
     * estimates via element-wise application of a `IProbabilityFunction`.
     */
    class LabelWiseProbabilityTransformation final : public IProbabilityTransformation {
        private:

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

        public:

            /**
             * @param probabilityFunctionPtr An unique pointer to an object of type `IProbabilityFunction` that should
             *                               be used to transform aggregated scores into probability estimates
             */
            LabelWiseProbabilityTransformation(std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr);

            void apply(CContiguousView<float64>::value_iterator scoresBegin,
                       CContiguousView<float64>::value_iterator scoresEnd) const override;
    };

}