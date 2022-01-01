/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_sparse.hpp"
#include "common/model/rule_list.hpp"


/**
 * Defines an interface for all classes that allow to predict whether individual labels of given query examples are
 * relevant or irrelevant using an existing rule-based model.
 */
class IClassificationPredictor : public ISparsePredictor<uint8> {

    public:

        virtual ~IClassificationPredictor() { };

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IClassificationPredictor`.
 */
class IClassificationPredictorFactory {

    public:

        virtual ~IClassificationPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IClassificationPredictor`.
         *
         * @param model A reference to an object of type `RuleList` that should be used to obtain predictions
         * @return      An unique pointer to an object of type `IClassificationPredictor` that has been created
         */
        virtual std::unique_ptr<IClassificationPredictor> create(const RuleList& model) const = 0;

};
