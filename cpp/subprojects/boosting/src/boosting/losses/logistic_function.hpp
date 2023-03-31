/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function_marginal.hpp"
#include "boosting/math/math.hpp"

namespace boosting {

    /**
     * Allows to transform regression scores that are predicted for individual labels into a marginal probability via
     * the logistic sigmoid function.
     */
    class LogisticFunction final : public IMarginalProbabilityFunction {
        public:

            float64 transformScoreIntoMarginalProbability(float64 score) const override {
                return logisticFunction(score);
            }
    };

    /**
     * Allows to create instances of the type `IMarginalProbabilityFunction` that transform regression scores that are
     * predicted for individual labels into a probability via the logistic sigmoid function.
     */
    class LogisticFunctionFactory final : public IMarginalProbabilityFunctionFactory {
        public:

            std::unique_ptr<IMarginalProbabilityFunction> create() const override {
                return std::make_unique<LogisticFunction>();
            }
    };

}
