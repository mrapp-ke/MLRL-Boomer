/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../head_refinement/prediction.h"
#include "model.h"
#include "condition_list.h"


/**
 * Defines an interface for all classes that allow to incrementally build rule-based models.
 */
class IModelBuilder {

    public:

        virtual ~IModelBuilder() { };

        /**
         * Initializes the model ans sets its default rule.
         *
         * This function must be called prior to the invocation of any other method of this class.
         *
         * @param prediction A pointer to an object of type `AbstractPrediction` that stores the scores that are
         *                   predicted by the default rule or a null pointer, if no default rule should be used
         */
        virtual void setDefaultRule(const AbstractPrediction* prediction) = 0;

        /**
         * Adds a new rule to the model.
         *
         * @param conditions    A reference to an object of type `ConditionList` that stores the rule's conditions
         * @param prediction    A reference to an object of type `AbstractPrediction` that stores the scores that are
         *                      predicted by the rule
         */
        virtual void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) = 0;

        /**
         * Builds and returns the model.
         *
         * @return An unique pointer to an object of type `IModel` that has been built
         */
        virtual std::unique_ptr<IModel> build() = 0;

};
