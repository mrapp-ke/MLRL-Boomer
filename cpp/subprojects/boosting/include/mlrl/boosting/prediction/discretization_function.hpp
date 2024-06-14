/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/prediction/probability_calibration_marginal.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to discretize scores.
     */
    class IDiscretizationFunction {
        public:

            virtual ~IDiscretizationFunction() {}

            /**
             * Discretizes the score that is predicted for a specific label.
             *
             * @param labelIndex    The index of the label, the score is predicted for
             * @param score         The score to be discretized
             * @return              A binary value the given score has been turned into
             */
            virtual bool discretizeScore(uint32 labelIndex, float64 score) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IDiscretizationFunction`.
     */
    class IDiscretizationFunctionFactory {
        public:

            virtual ~IDiscretizationFunctionFactory() {}

            /**
             * Creates and returns a new object of the type `IDiscretizationFunction`.
             *
             * @param marginalProbabilityCalibrationModel   A reference to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` that should be used
             *                                              for the calibration of marginal probabilities
             * @return                                      An unique pointer to an object of type
             *                                              `IDiscretizationFunction` that has been created
             */
            virtual std::unique_ptr<IDiscretizationFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;
    };

}
