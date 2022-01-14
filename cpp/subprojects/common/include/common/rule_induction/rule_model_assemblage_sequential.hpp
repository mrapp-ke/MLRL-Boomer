/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"


/**
 * Allows to configure an algorithm that sequentially induces several rules, optionally starting with a default rule,
 * that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageConfig final : public IRuleModelAssemblageConfig {

    private:

        bool useDefaultRule_;

    public:

        SequentialRuleModelAssemblageConfig();

        /**
         * Returns whether a default rule should be used or not.
         *
         * @return True, if a default rule should be used, false otherwise
         */
        bool getUseDefaultRule() const;

        /**
         * Sets whether a default rule should be used or not.
         *
         * @param useDefaultRule    True, if a default rule should be used, false otherwise
         * @return                  A reference to an object of type `SequentialRuleModelAssemblageConfig` that allows
         *                          further configuration of the algorithm for the induction of several rules that are
         *                          added to a rule-based model
         */
        SequentialRuleModelAssemblageConfig& setUseDefaultRule(bool useDefaultRule);

};

/**
 * A factory that allows to create instances of the class `IRuleModelAssemblage` that allow to sequentially induce
 * several rules, optionally starting with a default rule, that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageFactory final : public IRuleModelAssemblageFactory {

    private:

        bool useDefaultRule_;

    public:

        /**
         * @param useDefaultRule True, if a default rule should be used, false otherwise
         */
        SequentialRuleModelAssemblageFactory(bool useDefaultRule);

        std::unique_ptr<IRuleModelAssemblage> create(
            std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
            std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
            std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::unique_ptr<IPruningFactory> pruningFactoryPtr,
            std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
            std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const override;

};
