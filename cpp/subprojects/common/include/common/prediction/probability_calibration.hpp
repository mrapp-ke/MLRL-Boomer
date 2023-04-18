/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_vector.hpp"
#include "common/macros.hpp"
#include "common/statistics/statistics.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a model for the calibration of probabilities.
 */
class MLRLCOMMON_API IProbabilityCalibrationModel {
    public:

        virtual ~IProbabilityCalibrationModel() {};

        /**
         * Calibrates the marginal probability that is predicted for a specific label.
         *
         * @param labelIndex            The index of the label, the probability is predicted for
         * @param marginalProbability   The marginal probability to be calibrated
         * @return                      The calibrated probability
         */
        virtual float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const = 0;

        /**
         * Calibrates a joint probability.
         *
         * @param jointProbability  The joint probability to be calibrated
         * @return                  The calibrated probability
         */
        virtual float64 calibrateJointProbability(float64 jointProbability) const = 0;
};

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of probabilities.
 */
class IProbabilityCalibrator {
    public:

        virtual ~IProbabilityCalibrator() {};

        /**
         * Fits and returns a model for the calibration of probabilities.
         *
         * @param statistics A reference to an object of type `IStatistics` that provides access to statistics about the
         *                   labels of the training examples
         */
        virtual std::unique_ptr<IProbabilityCalibrationModel> fitCalibrationModel(
          const IStatistics& statistics) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * probabilities.
 */
class IProbabilityCalibratorConfig {
    public:

        virtual ~IProbabilityCalibratorConfig() {};

        /**
         * Creates and returns a new object of type `IProbabilityCalibrator` according to the configuration.
         *
         * @return An unique pointer to an object of type `IProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<IProbabilityCalibrator> createProbabilityCalibrator() const = 0;
};
