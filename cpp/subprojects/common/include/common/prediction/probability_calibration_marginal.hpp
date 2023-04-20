/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/probability_calibration.hpp"

/**
 * Defines an interface for all classes that allow to configure a method for fitting a model for the calibration of
 * marginal probabilities.
 */
class IMarginalProbabilityCalibratorConfig : public IProbabilityCalibratorConfig<IMarginalProbabilityCalibrator> {
    public:

        virtual ~IMarginalProbabilityCalibratorConfig() {};
};
