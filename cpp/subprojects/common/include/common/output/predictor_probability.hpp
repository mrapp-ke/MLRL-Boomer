/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor.hpp"


/**
 * Defines an interface for all classes that allow to predict label-wise probabilities for given query examples,
 * which estimate the chance of individual labels to be relevant, using an existing rule-based model.
 */
// TODO Add transform function
class IProbabilityPredictor : public IPredictor<float64> {

    public:

        virtual ~IProbabilityPredictor() { };

};
