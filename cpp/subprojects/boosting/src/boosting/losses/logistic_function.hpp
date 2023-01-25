/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function.hpp"
#include <cmath>


namespace boosting {

    /**
     * Calculates and returns the logistic function `1 / (1 + exp(-x))`, given a specific value `x`.
     *
     * This implementation exploits the identity `1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))` to increase numerical
     * stability (see, e.g., section "Numerically stable sigmoid function" in
     * https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
     *
     * @param x The value `x`
     * @return  The value that has been calculated
     */
    static inline constexpr float64 logisticFunction(float64 x) {
        if (x >= 0) {
            float64 exponential = std::exp(-x);  // Evaluates to 0 for large x, resulting in 1 ultimately
            return 1 / (1 + exponential);
        } else {
            float64 exponential = std::exp(x);  // Evaluates to 0 for large x, resulting in 0 ultimately
            return exponential / (1 + exponential);
        }
    }

    /**
     * Allows to transform the score that is predicted for an individual label into a probability by applying the
     * logistic sigmoid function.
     */
    class LogisticFunction final : public IProbabilityFunction {

        public:

            float64 transform(float64 predictedScore) const override {
                return logisticFunction(predictedScore);
            }

    };

    /**
     * Allows to create instances of the type `IProbabilityFunction` that transform the score that is predicted for an
     * individual label into a probability by applying the logistic sigmoid function.
     */
    class LogisticFunctionFactory final : public IProbabilityFunctionFactory {

        public:

            std::unique_ptr<IProbabilityFunction> create() const override {
                return std::make_unique<LogisticFunction>();
            }

    };

}
