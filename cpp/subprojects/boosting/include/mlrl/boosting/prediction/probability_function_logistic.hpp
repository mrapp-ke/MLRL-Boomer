/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/probability_function_marginal.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the type `IMarginalProbabilityFunction` that transform scores that are predicted
     * for individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunctionFactory final : public IMarginalProbabilityFunctionFactory {
        public:

            std::unique_ptr<IMarginalProbabilityFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;
    };

}
