/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor.hpp"
#include "common/model/rule_list.hpp"


/**
 * Defines an interface for all classes that allow predict label-wise regression scores for given query examples using
 * an existing rule-based model.
 */
class IRegressionPredictor : public IPredictor<float64> {

    public:

        virtual ~IRegressionPredictor() { };

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRegressionPredictor`.
 */
class IRegressionPredictorFactory {

    public:

        virtual ~IRegressionPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IRegressionPredictor`.
         *
         * @param model A reference to an object of type `RuleList` that should be used to obtain the predictions
         * @return      An unique pointer to an object of type `IRegressionPredictor` that has been created
         */
        virtual std::unique_ptr<IRegressionPredictor> create(const RuleList& model) const = 0;

};
