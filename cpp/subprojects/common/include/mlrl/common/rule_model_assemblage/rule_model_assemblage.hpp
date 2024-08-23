/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/model/model_builder.hpp"
#include "mlrl/common/rule_refinement/feature_space.hpp"
#include "mlrl/common/sampling/feature_sampling.hpp"
#include "mlrl/common/sampling/instance_sampling.hpp"
#include "mlrl/common/sampling/output_sampling.hpp"
#include "mlrl/common/sampling/partition_sampling.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"
#include "mlrl/common/stopping/stopping_criterion.hpp"

#include <memory>

/**
 * Defines an interface for all classes that implement an algorithm for the induction of several rules that will be
 * added to a rule-based model.
 */
class IRuleModelAssemblage {
    public:

        virtual ~IRuleModelAssemblage() {}

        /**
         * Assembles and returns a rule-based model that consists of several rules.
         *
         * @param partition             A reference to an object of type `IPartition` that provides access to the
         *                              indices of the training examples that belong to the training set and the holdout
         *                              set, respectively
         * @param outputSampling        A reference to an object of type `IOutputSampling` to be used for sampling the
         *                              outputs whenever a new rule is induced
         * @param instanceSampling      A reference to an object of type `IInstanceSampling` to be used for sampling the
         *                              examples whenever a new rule is induced
         * @param featureSampling       A reference to an object of type `IFeatureSampling` to be used for sampling the
         *                              features that may be used by the conditions of a rule
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              the statistics which serve as the basis for learning rules
         * @param featureSpace          A reference to an object of type `IFeatureSpace` that provides access to the
         *                              feature space
         * @param modelBuilder          A reference to an object of type `IModelBuilder`, the rules should be added to
         */
        virtual void induceRules(IPartition& partition, IOutputSampling& outputSampling,
                                 IInstanceSampling& instanceSampling, IFeatureSampling& featureSampling,
                                 IStatisticsProvider& statisticsProvider, IFeatureSpace& featureSpace,
                                 IModelBuilder& modelBuilder) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRuleModelAssemblage`.
 */
class IRuleModelAssemblageFactory {
    public:

        virtual ~IRuleModelAssemblageFactory() {}

        /**
         * Creates and returns a new object of the type `IRuleModelAssemblage`.
         *
         * @param stoppingCriterionFactoryPtr   An unique pointer to an object of type `IStoppingCriterionFactory` that
         *                                      allows to create the implementations to be used to decide whether
         *                                      additional rules should be induced or not
         */
        virtual std::unique_ptr<IRuleModelAssemblage> create(
          std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure an algorithm for the induction of several rules that
 * will be added to a rule-based model.
 */
class IRuleModelAssemblageConfig {
    public:

        virtual ~IRuleModelAssemblageConfig() {}

        /**
         * Creates and returns a new object of type `IRuleModelAssemblageFactory` according to specified configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param outputMatrix  A reference to an object of type `IOutputMatrix` that provides row-wise access to the
         *                      ground truth of the training examples
         * @return              An unique pointer to an object of type `IRuleModelAssemblageFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
          const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const = 0;
};
