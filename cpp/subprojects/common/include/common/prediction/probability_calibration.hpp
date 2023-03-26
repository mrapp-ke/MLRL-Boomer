/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_vector.hpp"

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
