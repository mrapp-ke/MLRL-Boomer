/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/prediction/probability_calibration_marginal.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform scores that are predicted for individual labels into
     * marginal probabilities.
     */
    class IMarginalProbabilityFunction {
        public:

            virtual ~IMarginalProbabilityFunction() {}

            /**
             * Transforms a score, represented by a 32-bit floating point value, that is predicted for a specific label
             * into a probability.
             *
             * @param labelIndex    The index of the label, the score is predicted for
             * @param score         The score that is predicted
             * @return              The probability into which the given score was transformed
             */
            virtual float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float32 score) const = 0;

            /**
             * Transforms a score, represented by a 64-bit floating point value, that is predicted for a specific label
             * into a probability.
             *
             * @param labelIndex    The index of the label, the score is predicted for
             * @param score         The score that is predicted
             * @return              The probability into which the given score was transformed
             */
            virtual float64 transformScoreIntoMarginalProbability(uint32 labelIndex, float64 score) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IMarginalProbabilityFunction`.
     */
    class IMarginalProbabilityFunctionFactory {
        public:

            virtual ~IMarginalProbabilityFunctionFactory() {}

            /**
             * Creates and returns a new object of the type `IMarginalProbabilityFunction`.
             *
             * @param marginalProbabilityCalibrationModel   A reference to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` that should be used
             *                                              for the calibration of marginal probabilities
             * @return                                      An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunction` that has been created
             */
            virtual std::unique_ptr<IMarginalProbabilityFunction> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;
    };

}
