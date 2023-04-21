/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration_isotonic.hpp"

namespace boosting {

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
     * Allows to configure a calibrator that fits a model for the calibration of joint probabilities via isotonic
     * regression.
     */
    class IsotonicJointProbabilityCalibratorConfig final : public IJointProbabilityCalibratorConfig {
        public:

            /**
             * @see `IJointProbabilityCalibratorConfig::createJointProbabilityCalibrator`
             */
            std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator() const override;
    };

}
