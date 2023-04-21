/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_joint.hpp"

/**
 * Defines an interface for all models for the calibration of marginal probabilities via isotonic regression.
 */
class MLRLCOMMON_API IIsotonicMarginalProbabilityCalibrationModel : public IMarginalProbabilityCalibrationModel {
    public:

        virtual ~IIsotonicMarginalProbabilityCalibrationModel() override {};
};

/**
 * A model for the calibration of marginal probabilities via isotonic regression.
 */
class IsotonicMarginalProbabilityCalibrationModel final : public IIsotonicMarginalProbabilityCalibrationModel {
    public:

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override;
};

/**
 * Creates and returns a new object of the type `IIsotonicMarginalProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `IIsotonicMarginalProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel>
  createIsotonicMarginalProbabilityCalibrationModel();

/**
 * Defines an interface for all model for the calibration of joint probabilities via isotonic regression.
 */
class MLRLCOMMON_API IIsotonicJointProbabilityCalibrationModel : public IJointProbabilityCalibrationModel {
    public:

        virtual ~IIsotonicJointProbabilityCalibrationModel() override {};
};

/**
 * A model for the calibration of joint probabilities via isotonic regression.
 */
class IsotonicJointProbabilityCalibrationModel final : public IIsotonicJointProbabilityCalibrationModel {
    public:

        float64 calibrateJointProbability(float64 jointProbability) const override;
};

/**
 * Creates and returns a new object of the type `IIsotonicJointProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `IIsotonicJointProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IIsotonicJointProbabilityCalibrationModel>
  createIsotonicJointProbabilityCalibrationModel();
