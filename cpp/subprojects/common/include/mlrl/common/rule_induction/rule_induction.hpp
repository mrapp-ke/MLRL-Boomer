/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_matrix.hpp"
#include "mlrl/common/input/output_matrix.hpp"
#include "mlrl/common/model/model_builder.hpp"
#include "mlrl/common/post_processing/post_processor.hpp"
#include "mlrl/common/rule_pruning/rule_pruning.hpp"
#include "mlrl/common/rule_refinement/feature_space.hpp"
#include "mlrl/common/sampling/feature_sampling.hpp"
#include "mlrl/common/sampling/partition.hpp"
#include "mlrl/common/sampling/weight_vector.hpp"
#include "mlrl/common/statistics/statistics.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement an algorithm for the induction of individual rules.
 */
class IRuleInduction {
    public:

        virtual ~IRuleInduction() {}

        /**
         * Induces the default rule.
         *
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which should serve as the basis for inducing the default rule
         * @param modelBuilder  A reference to an object of type `IModelBuilder`, the default rule should be added to
         */
        virtual void induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const = 0;

        /**
         * Induces a new rule.
         *
         * @param featureSpace      A reference to an object of type `IFeatureSpace` that provides access to the feature
         *                          space
         * @param outputIndices     A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the outputs for which the rule may predict
         * @param weights           A reference to an object of type `IWeightVector` that provides access to the weights
         *                          of individual training examples
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that belong to the training set and the holdout set,
         *                          respectively
         * @param featureSampling   A reference to an object of type `IFeatureSampling` that should be used for sampling
         *                          the features that may be used by a new condition
         * @param modelBuilder      A reference to an object of type `IModelBuilder`, the rule should be added to
         * @return                  True, if a rule has been induced, false otherwise
         */
        virtual bool induceRule(IFeatureSpace& featureSpace, const IIndexVector& outputIndices,
                                const IWeightVector& weights, IPartition& partition, IFeatureSampling& featureSampling,
                                IModelBuilder& modelBuilder) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRuleInduction`.
 */
class IRuleInductionFactory {
    public:

        virtual ~IRuleInductionFactory() {}

        /**
         * Creates and returns a new object of type `IRuleInduction`.
         *
         * @return An unique pointer to an object of type `IRuleInduction` that has been created.
         */
        virtual std::unique_ptr<IRuleInduction> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure an algorithm for the induction of individual rules.
 */
class IRuleInductionConfig {
    public:

        virtual ~IRuleInductionConfig() {}

        /**
         * Creates and returns a new object of type `IRuleInductionFactory` according to the specified configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides access to the ground
         *                      truth of the training examples
         * @return              An unique pointer to an object of type `IRuleInductionFactory` that has been created
         */
        virtual std::unique_ptr<IRuleInductionFactory> createRuleInductionFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const = 0;
};
