/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/macros.hpp"
#include "common/prediction/probability_calibration_marginal.hpp"

/**
 * Defines an interface for all models for the calibration of marginal probabilities via isotonic regression.
 */
class MLRLBOOSTING_API IIsotonicMarginalProbabilityCalibrationModel : public IMarginalProbabilityCalibrationModel {
    public:

        virtual ~IIsotonicMarginalProbabilityCalibrationModel() override {};
};

/**
 * Allows to configure a calibrator that fits a model for the calibration of marginal probabilities via isotonic
 * regression.
 */
class IsotonicMarginalProbabilityCalibratorConfig final : public IMarginalProbabilityCalibratorConfig {
    public:

        /**
         * @see `IMarginalProbabilityCalibratorConfig::createMarginalProbabilityCalibrator`
         */
        std::unique_ptr<IMarginalProbabilityCalibrator> createMarginalProbabilityCalibrator() const override;
};

/**
 * Creates and returns a new object of the type `IIsotonicMarginalProbabilityCalibrationModel`.
 *
 * @return An unique pointer to an object of type `IIsotonicMarginalProbabilityCalibrationModel` that has been created
 */
MLRLBOOSTING_API std::unique_ptr<IIsotonicMarginalProbabilityCalibrationModel>
  createIsotonicMarginalProbabilityCalibrationModel();
