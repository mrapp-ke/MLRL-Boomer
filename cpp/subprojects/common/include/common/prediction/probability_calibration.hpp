/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_vector.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement a model for the calibration of probabilities.
 */
class IProbabilityCalibrationModel {
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
 * Defines an interface for all classes that allow to create instances of the type `IProbabilityCalibrator`.
 */
class IProbabilityCalibratorFactory {
    public:

        virtual ~IProbabilityCalibratorFactory() {};

        /**
         * Creates and returns a new object of the type `IProbabilityCalibrator`.
         *
         * @return An unique pointer to an object of type `IProbabilityCalibrator` that has been created
         */
        virtual std::unique_ptr<IProbabilityCalibrator> create() const = 0;
};
