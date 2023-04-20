/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <memory>

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * probabilities.
 *
 * @tparam ProbabilityCalibrator The type of the method for fitting a calibration model
 */
template<typename ProbabilityCalibrator>
class IProbabilityCalibratorConfig {
    public:

        virtual ~IProbabilityCalibratorConfig() {};

        /**
         * Creates and returns a new object of template type `ProbabilityCalibrator` according to the configuration.
         *
         * @return An unique pointer to an object of template type `ProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<ProbabilityCalibrator> createProbabilityCalibrator() const = 0;
};
