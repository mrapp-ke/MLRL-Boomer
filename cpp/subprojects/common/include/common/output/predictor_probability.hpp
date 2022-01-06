/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor.hpp"
#include "common/output/label_vector_set.hpp"
#include "common/model/rule_list.hpp"


/**
 * Defines an interface for all classes that allow to predict label-wise probabilities for given query examples,
 * which estimate the chance of individual labels to be relevant, using an existing rule-based model.
 */
class IProbabilityPredictor : virtual public IPredictor<float64> {

    public:

        virtual ~IProbabilityPredictor() override { };

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
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> create(const RuleList& model,
                                                              const LabelVectorSet* labelVectorSet) const = 0;

};
