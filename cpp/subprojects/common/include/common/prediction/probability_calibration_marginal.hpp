/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/prediction/probability_calibration.hpp"

/**
 * Defines an interface for all classes that implement a model for the calibration of marginal probabilities.
 */
class MLRLCOMMON_API IMarginalProbabilityCalibrationModel {
    public:

        virtual ~IMarginalProbabilityCalibrationModel() {};

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
 * Defines an interface for all classes that implement a method for fitting models for the calibration of marginal
 * probabilities.
 */
class IMarginalProbabilityCalibrator : public IProbabilityCalibrator<IMarginalProbabilityCalibrationModel> {
    public:

        virtual ~IMarginalProbabilityCalibrator() override {};
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * marginal probabilities.
 */
class IMarginalProbabilityCalibratorConfig : public IProbabilityCalibratorConfig<IMarginalProbabilityCalibrator> {
    public:

        virtual ~IMarginalProbabilityCalibratorConfig() override {};
};
