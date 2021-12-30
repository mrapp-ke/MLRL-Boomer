/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/body.hpp"
#include "common/model/head.hpp"
#include <memory>

// Forward declarations
class IClassificationPredictorFactory;
class IClassificationPredictor;
class IRegressionPredictorFactory;
class IRegressionPredictor;
class IProbabilityPredictorFactory;
class IProbabilityPredictor;


/**
 * Defines an interface for all rule-based models.
 */
class IRuleModel {

    public:

        virtual ~IRuleModel() { };

        /**
         * Returns the total number of rules in the model.
         *
         * @return The number of rules
         */
        virtual uint32 getNumRules() const = 0;

        /**
         * Returns the number of used rules.
         *
         * @return The number of used rules
         */
        virtual uint32 getNumUsedRules() const = 0;

        /**
         * Sets the number of used rules.
         *
         * @param numUsedRules The number of used rules to be set or 0, if all rules are used
         */
        virtual void setNumUsedRules(uint32 numUsedRules) = 0;

        /**
         * Creates a new rule from a given body and head and adds it to the model.
         *
         * @param bodyPtr An unique pointer to an object of type `IBody` that should be used as the body of the rule
         * @param headPtr An unique pointer to an object of type `IHead` that should be used as the head of the rule
         */
        virtual void addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) = 0;

        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the bodies and heads
         * of the rules that are contained in this model.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param completeHeadVisitor       The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        virtual void visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                           IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                           IHead::CompleteHeadVisitor completeHeadVisitor,
                           IHead::PartialHeadVisitor partialHeadVisitor) const = 0;


        /**
         * Invokes some of the given visitor functions, depending on which ones are able to handle the bodies and heads
         * of the used rules that are contained in this model.
         *
         * @param emptyBodyVisitor          The visitor function for handling objects of the type `EmptyBody`
         * @param conjunctiveBodyVisitor    The visitor function for handling objects of the type `ConjunctiveBody`
         * @param completeHeadVisitor       The visitor function for handling objects of the type `CompleteHead`
         * @param partialHeadVisitor        The visitor function for handling objects of the type `PartialHead`
         */
        virtual void visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor,
                               IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                               IHead::CompleteHeadVisitor completeHeadVisitor,
                               IHead::PartialHeadVisitor partialHeadVisitor) const = 0;

        /**
         * Creates and returns a new instance of the class `IClassificationPredictor`, based on the type of this
         * rule-based model.
         *
         * @param factory   A reference to an object of type `IClassificationPredictorFactory` that should be used to
         *                  create the instance
         * @return          An unique pointer to an object of type `IClassificationPredictor` that has been created
         */
        virtual std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
            const IClassificationPredictorFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IRegressionPredictor`, based on the type of this rule-based
         * model.
         *
         * @param factory   A reference to an object of type `IRegressionPredictorFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IRegressionPredictor` that has been created
         */
        virtual std::unique_ptr<IRegressionPredictor> createRegressionPredictor(
            const IRegressionPredictorFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this rule-based
         * model.
         *
         * @param factory   A reference to an object of type `IProbabilityPredictorFactory` that should be used to
         *                  create the instance
         * @return          An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory) const = 0;

};
