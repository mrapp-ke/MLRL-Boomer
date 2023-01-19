/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/macros.hpp"
#include <memory>

// Forward declarations
class IClassificationPredictorFactory;
class IClassificationPredictor;
class IRegressionPredictorFactory;
class IOldRegressionPredictor;
class IProbabilityPredictorFactory;
class IOldProbabilityPredictor;
class ILabelSpaceInfo;


/**
 * Defines an interface for all rule-based models.
 */
class MLRLCOMMON_API IRuleModel {

    public:

        virtual ~IRuleModel() { };

        /**
         * Returns the total number of rules in the model, including the default rule, if available.
         *
         * @return The number of rules
         */
        virtual uint32 getNumRules() const = 0;

        /**
         * Returns the number of used rules, including the default rule, if available.
         *
         * @return The number of used rules
         */
        virtual uint32 getNumUsedRules() const = 0;

        /**
         * Sets the number of used rules, including the default rule, if available.
         *
         * @param numUsedRules The number of used rules to be set or 0, if all rules should be used
         */
        virtual void setNumUsedRules(uint32 numUsedRules) = 0;

        /**
         * Creates and returns a new instance of the class `IClassificationPredictor`, based on the type of this
         * rule-based model.
         *
         * @param factory           A reference to an object of type `IClassificationPredictorFactory` that should be
         *                          used to create the instance
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for making predictions
         * @return                  An unique pointer to an object of type `IClassificationPredictor` that has been
         *                          created
         */
        virtual std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
            const IClassificationPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const = 0;

        /**
         * Creates and returns a new instance of the class `IRegressionPredictor`, based on the type of this rule-based
         * model.
         *
         * @param factory           A reference to an object of type `IRegressionPredictorFactory` that should be used
         *                          to create the instance
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for making predictions
         * @return                  An unique pointer to an object of type `IRegressionPredictor` that has been created
         */
        virtual std::unique_ptr<IOldRegressionPredictor> createRegressionPredictor(
            const IRegressionPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this rule-based
         * model.
         *
         * @param factory           A reference to an object of type `IProbabilityPredictorFactory` that should be used
         *                          to create the instance
         * @param labelSpaceInfo    A reference to an object of type `ILabelSpaceInfo` that provides information about
         *                          the label space that may be used as a basis for making predictions
         * @return                  An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IOldProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const = 0;

};
