/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/model_builder.hpp"
#include "common/post_processing/post_processor.hpp"
#include "common/pruning/pruning.hpp"
#include "common/rule_induction/rule_induction.hpp"
#include "common/sampling/feature_sampling.hpp"
#include "common/sampling/label_sampling.hpp"
#include "common/thresholds/thresholds.hpp"


// TODO comment
class IPostOptimizationPhase {

    public:

        virtual ~IPostOptimizationPhase() { };

        // TODO optimize method

};

// TODO comment.
class IPostOptimizationPhaseFactory {

    public:

        virtual ~IPostOptimizationPhaseFactory() { };

        // TODO Comment
        virtual std::unique_ptr<IPostOptimizationPhase> create() const = 0;

};

/**
 * Defines an interface for all classes that allow to optimize a rule-based model globally once it has been learned by
 * carrying out several optimization phases.
 */
class IPostOptimization {

    public:

        virtual ~IPostOptimization() { };

        // TODO Comment
        virtual IModelBuilder& getModelBuilder() const = 0;

        /**
         * Optimizes a rule-based model globally once it has been learned by carrying out several optimization phases.
         *
         * @param thresholds        A reference to an object of type `IThresholds` that provides access to the
         *                          thresholds that may be used by the conditions of the rule
         * @param ruleInduction     A reference to an object of type `IRuleInduction` that should be used for inducing
         *                          new rules
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that belong to the training set and the holdout set,
         *                          respectively
         * @param labelSampling     A reference to an object of type `ILabelSampling` that should be used for sampling
         *                          labels
         * @param instanceSampling  A reference to an object of type `IInstanceSampling` that should be used for
         *                          sampling examples
         * @param featureSampling   A reference to an object of type `IFeatureSampling` that should be used for sampling
         *                          the features that may be used by the conditions of new rules
         * @param pruning           A reference to an object of type `IPruning` that should be used to prune new rules
         * @param postProcessor     A reference to an object of type `IPostProcessor` that should be used to
         *                          post-process the predictions of new rules
         * @param rng               A reference to an object of type `RNG` that implements the random number generator
         *                          to be used
         */
        virtual void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                                   ILabelSampling& labelSampling, IInstanceSampling& instanceSampling,
                                   IFeatureSampling& featureSampling, const IPruning& pruning,
                                   const IPostProcessor& postProcessor, RNG& rng) = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IPostOptimization`.
 */
class IPostOptimizationFactory {

    public:

        virtual ~IPostOptimizationFactory() { };

        /**
         * Creates and returns a new object of type `PostOptimization`.
         *
         * @param modelBuilderFactory   A reference to an object of type `IModelBuilderFactory` that allows to create
         *                              the builder to be used for assembling a model
         * @return                      An unique pointer to an object of type `IPostOptimization` that has been created
         */
        virtual std::unique_ptr<IPostOptimization> create(const IModelBuilderFactory& modelBuilderFactory) const = 0;

};

/**
 * Defines an interface for all classes that allow to configure a method that optimizes a rule-based model globally once
 * it has been learned.
 */
class IPostOptimizationConfig {

    public:

        virtual ~IPostOptimizationConfig() { };

        /**
         * Creates and returns a new object of type `IPostOptimizationFactory` according to the specified configuration.
         *
         * @return An unique pointer to an object of type `IPostOptimizationFactory` that has been created
         */
        virtual std::unique_ptr<IPostOptimizationFactory> createPostOptimizationFactory() const = 0;

};
