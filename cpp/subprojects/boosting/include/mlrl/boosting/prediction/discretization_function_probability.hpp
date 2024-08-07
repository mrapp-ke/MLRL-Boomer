/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/discretization_function.hpp"
#include "mlrl/boosting/prediction/probability_function_marginal.hpp"

#include <memory>

namespace boosting {

    /**
     * Allow to create instances of the type `IDiscretizationFunction` that discretize scores by transforming them into
     * marginal probabilities.
     */
    class ProbabilityDiscretizationFunctionFactory : public IDiscretizationFunctionFactory {
        private:

            std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              the implementation to be used to transform scores into
             *                                              marginal probabilities
             */
            ProbabilityDiscretizationFunctionFactory(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr);

            std::unique_ptr<IDiscretizationFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;
    };

}
