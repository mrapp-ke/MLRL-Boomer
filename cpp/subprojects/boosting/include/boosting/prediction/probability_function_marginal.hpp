/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform regression scores that are predicted for individual
     * labels into marginal probabilities.
     */
    class IMarginalProbabilityFunction {
        public:

            virtual ~IMarginalProbabilityFunction() {};

            /**
             * Transforms the regression score that is predicted for a specific label into a probability.
             *
             * @param labelIndex    The index of the label, the regression score is predicted for
             * @param score         The regression score that is predicted
             * @return              The probability into which the given score was transformed
             */
            virtual float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IMarginalProbabilityFunction`.
     */
    class IMarginalProbabilityFunctionFactory {
        public:

            virtual ~IMarginalProbabilityFunctionFactory() {};

            /**
             * Creates and returns a new object of the type `IMarginalProbabilityFunction`.
             *
             * @param probabilityCalibrationModel   A reference to an object of type `IProbabilityCalibrationModel` that
             *                                      should be used for the calibration of probabilities
             * @return                              An unique pointer to an object of type
             *                                      `IMarginalProbabilityFunction` that has been created
             */
            virtual std::unique_ptr<IMarginalProbabilityFunction> create(
              const IProbabilityCalibrationModel& probabilityCalibrationModel) const = 0;
    };

}
