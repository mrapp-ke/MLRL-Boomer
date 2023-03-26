/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_vector.hpp"
#include "common/macros.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a model for the calibration of probabilities.
 */
class MLRLCOMMON_API IProbabilityCalibrationModel {
    public:

        virtual ~IProbabilityCalibrationModel() {};

        /**
         * Calibrates given probabilities.
         *
         * @param probabilitiesBegin
         * @param probabilitiesEnd
         */
        virtual void calibrateProbabilities(VectorView<float64>::iterator probabilitiesBegin,
                                            VectorView<float64>::iterator probabilitiesEnd) const = 0;
};

/**
 * Defines an interface for all classes that implement a method for fitting models for the calibration of probabilities.
 */
class IProbabilityCalibrator {
    public:

        virtual ~IProbabilityCalibrator() {};

        /**
         * Fits and returns a model for the calibration of probabilities.
         */
        virtual std::unique_ptr<IProbabilityCalibrationModel> fitCalibrationModel() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * probabilities.
 */
class IProbabilityCalibratorConfig {
    public:

        virtual ~IProbabilityCalibratorConfig() {};

        /**
         * Creates and returns a new object of type `IProbabilityCalibrator` according to the configuration.
         *
         * @return An unique pointer to an object of type `IProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<IProbabilityCalibrator> createProbabilityCalibrator() const = 0;
};
