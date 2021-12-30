/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor.hpp"
#include "common/model/rule_model.hpp"


/**
 * Defines an interface for all classes that allow to predict label-wise probabilities for given query examples,
 * which estimate the chance of individual labels to be relevant, using an existing rule-based model.
 */
// TODO Add transform function
class IProbabilityPredictor : public IPredictor<float64> {

    public:

        virtual ~IProbabilityPredictor() { };

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IProbabilityPredictor`.
 */
class IProbabilityPredictorFactory {

    public:

        virtual ~IProbabilityPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IProbabilityPredictor`.
         *
         * @param model A reference to an object of type `RuleModel` that should be used to obtain the predictions
         * @return      An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> create(const RuleModel& model) const = 0;

};
