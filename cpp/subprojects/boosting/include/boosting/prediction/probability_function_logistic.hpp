/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function_marginal.hpp"

namespace boosting {

    /**
     * An implementation of the class `IMarginalProbabilityFunction` that transforms regression scores that are
     * predicted for individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunction final : public IMarginalProbabilityFunction {
        public:

            float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const override;
    };

    /**
     * Allows to create instances of the type `IMarginalProbabilityFunction` that transform regression scores that are
     * predicted for individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunctionFactory final : public IMarginalProbabilityFunctionFactory {
        public:

            std::unique_ptr<IMarginalProbabilityFunction> create() const override;
    };

}
