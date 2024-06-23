/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/prediction/discretization_function.hpp"

#include <memory>

namespace boosting {

    /**
     * Allow to create instances of the type `IDiscretizationFunction` that discretize scores by comparing them to a
     * threshold.
     */
    class ScoreDiscretizationFunctionFactory : public IDiscretizationFunctionFactory {
        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold that should be used for discretization
             */
            explicit ScoreDiscretizationFunctionFactory(float64 threshold);

            std::unique_ptr<IDiscretizationFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;
    };

}
