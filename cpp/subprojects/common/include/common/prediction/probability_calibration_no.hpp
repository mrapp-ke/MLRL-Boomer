/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_marginal.hpp"

/**
 * Defines an interface for all models for the calibration of probabilities that do make any adjustments.
 */
class MLRLCOMMON_API INoProbabilityCalibrationModel : public IMarginalProbabilityCalibrationModel {
    public:

        virtual ~INoProbabilityCalibrationModel() override {};
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of probabilities.
 */
class NoProbabilityCalibratorConfig final : public IMarginalProbabilityCalibratorConfig {
    public:

        std::unique_ptr<IMarginalProbabilityCalibrator> createProbabilityCalibrator() const override;
};

/**
 * Creates and returns a new object of the type `INoProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `INoProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoProbabilityCalibrationModel> createNoProbabilityCalibrationModel();
