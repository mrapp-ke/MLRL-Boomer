/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/probability_function_joint.hpp"
#include "mlrl/boosting/prediction/probability_function_marginal.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the type `IJointProbabilityFunction` that transform scores that are predicted for
     * an example into joint probabilities by applying an `IMarginalProbabilityFunction` to each one and calculating the
     * product of the resulting marginal probabilities according to the probabilistic chain rule.
     */
    class ChainRuleFactory final : public IJointProbabilityFunctionFactory {
        private:

            std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              implementations of the function to be used to transform
             *                                              scores into marginal probabilities
             */
            ChainRuleFactory(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr);

            std::unique_ptr<IJointProbabilityFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override;
    };

}
