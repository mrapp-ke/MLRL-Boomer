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
        private:

            const IProbabilityCalibrationModel& probabilityCalibrationModel_;

        public:

            /**
             * @param probabilityCalibrationModel A reference to an object of type `IProbabilityCalibrationModel` that
             *                                    should be used for the calibration of probabilities
             */
            LogisticFunction(const IProbabilityCalibrationModel& probabilityCalibrationModel);

            float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const override;
    };

    /**
     * Allows to create instances of the type `IMarginalProbabilityFunction` that transform regression scores that are
     * predicted for individual labels into marginal probabilities via the logistic sigmoid function.
     */
    class LogisticFunctionFactory final : public IMarginalProbabilityFunctionFactory {
        public:

            std::unique_ptr<IMarginalProbabilityFunction> create(
              const IProbabilityCalibrationModel& probabilityCalibrationModel) const override;
    };

}
