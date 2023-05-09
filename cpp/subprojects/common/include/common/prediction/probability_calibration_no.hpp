/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_joint.hpp"

/**
 * Defines an interface for all models for the calibration of marginal probabilities that do make any adjustments.
 */
class MLRLCOMMON_API INoMarginalProbabilityCalibrationModel : public IMarginalProbabilityCalibrationModel {
    public:

        virtual ~INoMarginalProbabilityCalibrationModel() override {};
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of marginal probabilities.
 */
class NoMarginalProbabilityCalibratorConfig final : public IMarginalProbabilityCalibratorConfig {
    public:

        bool shouldUseHoldoutSet() const override;

        std::unique_ptr<IMarginalProbabilityCalibrator> createMarginalProbabilityCalibrator() const override;
};

/**
 * Creates and returns a new object of the type `INoMarginalProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `INoMarginalProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoMarginalProbabilityCalibrationModel> createNoMarginalProbabilityCalibrationModel();

/**
 * Defines an interface for all models for the calibration of joint probabilities that do make any adjustments.
 */
class MLRLCOMMON_API INoJointProbabilityCalibrationModel : public IJointProbabilityCalibrationModel {
    public:

        virtual ~INoJointProbabilityCalibrationModel() override {};
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of joint probabilities.
 */
class NoJointProbabilityCalibratorConfig final : public IJointProbabilityCalibratorConfig {
    public:

        std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator() const override;
};

/**
 * Creates and returns a new object of the type `INoJointProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `INoJointProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoJointProbabilityCalibrationModel> createNoJointProbabilityCalibrationModel();
