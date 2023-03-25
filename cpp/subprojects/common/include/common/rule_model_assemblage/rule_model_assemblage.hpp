/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_info.hpp"
#include "common/input/feature_matrix_column_wise.hpp"
#include "common/input/label_matrix_row_wise.hpp"
#include "common/model/model_builder.hpp"
#include "common/rule_induction/rule_induction.hpp"
#include "common/sampling/feature_sampling.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/label_sampling.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/statistics/statistics_provider.hpp"
#include "common/stopping/stopping_criterion.hpp"
#include "common/thresholds/thresholds.hpp"

/**
 * Defines an interface for all classes that implement an algorithm for the induction of several rules that will be
 * added to a rule-based model.
 */
class IRuleModelAssemblage {
    public:

        virtual ~IRuleModelAssemblage() {};

        /**
         * Assembles and returns a rule-based model that consists of several rules.
         *
         * @param featureInfo           A reference to an object of type `IFeatureInfo` that provides information about
         *                              the types of individual features
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of individual training examples
         * @param labelMatrix           A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise
         *                              access to the labels of individual training examples
         * @param ruleInduction         A reference to an object of type `IRuleInduction` to be used for the induction
         *                              of individual rules
         * @param partition             A reference to an object of type `IPartition` that provides access to the
         *                              indices of the training examples that belong to the training set and the holdout
         *                              set, respectively
         * @param labelSampling         A reference to an object of type `ILabelSampling` to be used for sampling the
         *                              labels whenever a new rule is induced
         * @param instanceSampling      A reference to an object of type `IInstanceSampling` to be used for sampling the
         *                              examples whenever a new rule is induced
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              the statistics which serve as the basis for learning rules
         * @param thresholds            A reference to an object of type `IThresholds` that provides access to the
         *                              thresholds that may be used by the conditions of rules
         * @param rng                   A reference to an object of type `RNG` that implements the random number
         *                              generator to be used
         * @param modelBuilder          A reference to an object of type `IModelBuilder`, the rules should be added to
         */
        virtual void induceRules(const IFeatureInfo& featureInfo, const IColumnWiseFeatureMatrix& featureMatrix,
                                 const IRowWiseLabelMatrix& labelMatrix, const IRuleInduction& ruleInduction,
                                 IPartition& partition, ILabelSampling& labelSampling,
                                 IInstanceSampling& instanceSampling, IStatisticsProvider& statisticsProvider,
                                 IThresholds& thresholds, IModelBuilder& modelBuilder, RNG& rng) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRuleModelAssemblage`.
 */
class IRuleModelAssemblageFactory {
    public:

        virtual ~IRuleModelAssemblageFactory() {};

        /**
         * Creates and returns a new object of the type `IRuleModelAssemblage`.
         *
         * @param featureSamplingFactoryPtr     An unique pointer to an object of type `IFeatureSamplingFactory` that
         *                                      allows to create the implementation to be used for sampling the features
         *                                      that may be used by the conditions of a rule
         * @param rulePruningFactoryPtr         An unique pointer to an object of type `IRulePruningFactory` that allows
         *                                      to create the implementation to be used for pruning rules
         * @param postProcessorFactoryPtr       An unique pointer to an object of type `IPostProcessorFactory` that
         *                                      allows to create the implementation to be used for post-processing the
         *                                      predictions of rules
         * @param stoppingCriterionFactoryPtr   An unique pointer to an object of type `IStoppingCriterionFactory` that
         *                                      allows to create the implementations to be used to decide whether
         *                                      additional rules should be induced or not
         */
        virtual std::unique_ptr<IRuleModelAssemblage> create(
          std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
          std::unique_ptr<IRulePruningFactory> rulePruningFactoryPtr,
          std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
          std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure an algorithm for the induction of several rules that
 * will be added to a rule-based model.
 */
class IRuleModelAssemblageConfig {
    public:

        virtual ~IRuleModelAssemblageConfig() {};

        /**
         * Creates and returns a new object of type `IRuleModelAssemblageFactory` according to specified configuration.
         *
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IRuleModelAssemblageFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
          const IRowWiseLabelMatrix& labelMatrix) const = 0;
};
