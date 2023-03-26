/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration.hpp"

/**
 * Defines an interface for all models for the calibration of probabilities that do perform any adjustments.
 */
class MLRLCOMMON_API INoProbabilityCalibrationModel : public IProbabilityCalibrationModel {
    public:

        virtual ~INoProbabilityCalibrationModel() override {};
};

/**
 * Allows to configure a calibrator that does not fit a model for the calibration of probabilities.
 */
class NoProbabilityCalibratorConfig final : public IProbabilityCalibratorConfig {
    public:

        std::unique_ptr<IProbabilityCalibratorFactory> createProbabilityCalibratorFactory() const override;
};

/**
 * Creates and returns a new object of the type `INoProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `INoProbabilityCalibrationModel` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoProbabilityCalibrationModel> createNoProbabilityCalibrationModel();
