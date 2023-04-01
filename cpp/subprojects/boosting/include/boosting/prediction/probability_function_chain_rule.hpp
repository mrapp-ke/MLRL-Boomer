/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/probability_function_joint.hpp"
#include "boosting/prediction/probability_function_marginal.hpp"

namespace boosting {

    /**
     * An implementation of the class `IJointProbabilityFunction` that transforms regression scores that are
     * predicted for an example into joint probabilities by applying an `IMarginalProbabilityFunction` to each one and
     * calculating the product of the resulting marginal probabilities according to the probabilistic chain rule.
     */
    class ChainRule final : public IJointProbabilityFunction {
        private:

            const std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr_;

        public:

            /**
             * @param marginalProbabilityFunctionPtr An unique pointer to an object of type
             *                                       `IMarginalProbabilityFunction` to be used to transform regression
             *                                       scores into marginal probabilities
             */
            ChainRule(std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr);

            float64 transformScoresIntoJointProbability(VectorConstView<float64>::const_iterator scoresBegin,
                                                        VectorConstView<float64>::const_iterator scoresEnd,
                                                        const LabelVector& labelVector) const override;
    };

    /**
     * Allows to create instances of the type `IJointProbabilityFunction` that transform regression scores that are
     * predicted for an example into joint probabilities by applying an `IMarginalProbabilityFunction` to each one and
     * calculating the product of the resulting marginal probabilities according to the probabilistic chain rule.
     */
    class ChainRuleFactory final : public IJointProbabilityFunctionFactory {
        private:

            std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              implementations of the function to be used to transform
             *                                              regression scores into marginal probabilities
             */
            ChainRuleFactory(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr);

            std::unique_ptr<IJointProbabilityFunction> create() const override;
    };

}
