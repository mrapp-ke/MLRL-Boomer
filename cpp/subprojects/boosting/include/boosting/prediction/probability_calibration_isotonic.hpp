/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_isotonic.hpp"

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