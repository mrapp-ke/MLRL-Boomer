/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"


/**
 * A factory that allows to create instances of the class `SequentialRuleModelAssemblage`.
 */
class SequentialRuleModelAssemblageFactory final : public IRuleModelAssemblageFactory {

    public:

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
            std::forward_list<std::unique_ptr<IStoppingCriterionFactory>> stoppingCriterionFactories,
            bool useDefaultRule) const override;

};
