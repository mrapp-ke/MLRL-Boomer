/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/prediction/probability_calibration.hpp"

/**
 * Defines an interface for all classes that implement a model for the calibration of joint probabilities.
 */
class MLRLCOMMON_API IJointProbabilityCalibrationModel {
    public:

        virtual ~IJointProbabilityCalibrationModel() {};

        /**
         * Calibrates a joint probability.
         *
         * @param jointProbability  The joint probability to be calibrated
         * @return                  The calibrated probability
         */
        virtual float64 calibrateJointProbability(float64 jointProbability) const = 0;
};

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of joint
 * probabilities.
 */
class IJointProbabilityCalibrator : public IProbabilityCalibrator<IJointProbabilityCalibrationModel> {
    public:

        virtual ~IJointProbabilityCalibrator() override {};
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * joint probabilities.
 */
class IJointProbabilityCalibratorConfig : public IProbabilityCalibratorConfig<IJointProbabilityCalibrator> {
    public:

        virtual ~IJointProbabilityCalibratorConfig() override {};
};
