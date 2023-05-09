/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/macros.hpp"
#include "common/prediction/probability_calibration_isotonic.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a calibrator that fits a model for the calibration
     * of marginal probabilities via isotonic regression.
     */
    class MLRLBOOSTING_API IIsotonicMarginalProbabilityCalibratorConfig {
        public:

            virtual ~IIsotonicMarginalProbabilityCalibratorConfig() {};

            /**
             * Returns whether the calibration model is fit to the examples in the holdout set, if available, or not.
             *
             * @return True, if the calibration model is fit to the examples in the holdout set, if available, false
             *         if the training set is used instead
             */
            virtual bool isHoldoutSetUsed() const = 0;

            /**
             * Sets whether the calibration model should be fit to the examples in the holdout set, if available, or
             * not.
             *
             * @param useHoldoutSet True, if the calibration model should be fit to the examples in the holdout set, if
             *                      available, false if the training set should be used instead
             * @return              A reference to an object of type `IIsotonicMarginalProbabilityCalibratorConfig` that
             *                      allows further configuration of the calibrator
             */
            virtual IIsotonicMarginalProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet) = 0;
    };

    /**
     * Allows to configure a calibrator that fits a model for the calibration of marginal probabilities via isotonic
     * regression.
     */
    class IsotonicMarginalProbabilityCalibratorConfig final : public IIsotonicMarginalProbabilityCalibratorConfig,
                                                              public IMarginalProbabilityCalibratorConfig {
        private:

            bool useHoldoutSet_;

        public:

            IsotonicMarginalProbabilityCalibratorConfig();

            bool isHoldoutSetUsed() const override;

            IIsotonicMarginalProbabilityCalibratorConfig& setUseHoldoutSet(bool useHoldoutSet) override;

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
