/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/post_optimization/model_builder_intermediate.hpp"
#include "mlrl/common/rule_refinement/feature_space.hpp"
#include "mlrl/common/sampling/feature_sampling.hpp"
#include "mlrl/common/sampling/output_sampling.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to optimize a rule-based model globally once it has been learned.
 */
class IPostOptimizationPhase {
    public:

        virtual ~IPostOptimizationPhase() {}

        /**
         * Optimizes a rule-based model globally once it has been learned.
         *
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that belong to the training set and the holdout set,
         *                          respectively
         * @param outputSampling    A reference to an object of type `IOutputSampling` that should be used for sampling
         *                          outputs
         * @param instanceSampling  A reference to an object of type `IInstanceSampling` that should be used for
         *                          sampling examples
         * @param featureSampling   A reference to an object of type `IFeatureSampling` that should be used for sampling
         *                          the features that may be used by the conditions of new rules
         * @param featureSpace      A reference to an object of type `IFeatureSpace` that provides access to the feature
         *                          space
         */
        virtual void optimizeModel(IPartition& partition, IOutputSampling& outputSampling,
                                   IInstanceSampling& instanceSampling, IFeatureSampling& featureSampling,
                                   IFeatureSpace& featureSpace) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IPostOptimizationPhase`.
 */
class IPostOptimizationPhaseFactory {
    public:

        virtual ~IPostOptimizationPhaseFactory() {}

        /**
         * Creates and returns a new object of type `IPostOptimizationPhase`.
         *
         * @param modelBuilder  A reference to an object of type `IntermediateModelBuilder` that provides access to the
         *                      rules in the model
         * @return              An unique pointer to an object of type `IPostOptimizationPhase` that has been created
         */
        virtual std::unique_ptr<IPostOptimizationPhase> create(IntermediateModelBuilder& modelBuilder) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method that optimizes a rule-based model globally once
 * it has been learned.
 */
class IPostOptimizationPhaseConfig {
    public:

        virtual ~IPostOptimizationPhaseConfig() {}

        /**
         * Creates and returns a new object of type `IPostOptimizationPhaseFactory` according to the specified
         * configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides access to the ground
         *                      truth of the training examples
         * @return              An unique pointer to an object of type `IPostOptimizationPhaseFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IPostOptimizationPhaseFactory> createPostOptimizationPhaseFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const = 0;
};

/**
 * Defines an interface for all classes that allow to optimize a rule-based model globally once it has been learned by
 * carrying out several optimization phases.
 */
class IPostOptimization : public IPostOptimizationPhase {
    public:

        virtual ~IPostOptimization() override {}

        /**
         * Returns an `IModelBuilder` that is suited for post-optimization via this object. Rules that are induced
         * during training must be added to the returned builder.
         *
         * @return A reference to an object of type `IModelBuilder` that is suited for post-optimization
         */
        virtual IModelBuilder& getModelBuilder() const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IPostOptimization`.
 */
class IPostOptimizationFactory {
    public:

        virtual ~IPostOptimizationFactory() {}

        /**
         * Creates and returns a new object of type `IPostOptimization`.
         *
         * @param modelBuilderFactory   A reference to an object of type `IModelBuilderFactory` that allows to create
         *                              the builder to be used for assembling a model
         * @return                      An unique pointer to an object of type `IPostOptimization` that has been created
         */
        virtual std::unique_ptr<IPostOptimization> create(const IModelBuilderFactory& modelBuilderFactory) const = 0;
};
