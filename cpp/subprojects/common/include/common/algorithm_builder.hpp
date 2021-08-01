/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"


/**
 * TODO
 */
class AlgorithmBuilder final {

    private:

        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr_;

        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr_;

        std::shared_ptr<IRuleInduction> ruleInductionPtr_;

        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr_;

        std::shared_ptr<IHeadRefinementFactory> regularRuleHeadRefinementFactoryPtr_;

        std::shared_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr_;

        std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr_;

        std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr_;

        std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr_;

        std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr_;

        std::shared_ptr<IPruning> pruningPtr_;

        std::shared_ptr<IPostProcessor> postProcessorPtr_;

        std::forward_list<std::shared_ptr<IStoppingCriterion>> stoppingCriteria_;

    public:

        /**
         * @param statisticsProviderFactoryPtr  TODO
         * @param thresholdsFactoryPtr          TODO
         * @param ruleInductionPtr              TODO
         * @param headRefinementFactoryPtr      TODO
         * @param ruleModelAssemblageFactoryPtr TODO
         */
        AlgorithmBuilder(std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
                         std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr,
                         std::shared_ptr<IRuleInduction> ruleInductionPtr,
                         std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                         std::shared_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr);

        /**
         * TODO
         *
         * @param headRefinementFactoryPTr  TODO
         * @return                          TODO
         */
        AlgorithmBuilder& setDefaultRuleHeadRefinementFactory(
            std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr);

        /**
         * TODO
         *
         * @param labelSamplingFactoryPtr   TODO
         * @return                          TODO
         */
        AlgorithmBuilder& setLabelSamplingFactory(std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr);

        /**
         * TODO
         *
         * @param instanceSamplingFactoryPtr    TODO
         * @return                              TODO
         */
        AlgorithmBuilder& setInstanceSamplingFactory(
            std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr);

        /**
         * TODO
         *
         * @param featureSamplingFactoryPtr TODO
         * @return                          TODO
         */
        AlgorithmBuilder& setFeatureSamplingFactory(std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr);

        /**
         * TODO
         *
         * @param partitionSamplingFactoryPtr   TODO
         * @return                              TODO
         */
        AlgorithmBuilder& setPartitionSamplingFactory(
            std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr);

        /**
         * TODO
         *
         * @param pruningPtr    TODO
         * @return              TODO
         */
        AlgorithmBuilder& setPruning(std::unique_ptr<IPruning> pruningPtr);

        /**
         * TODO
         *
         * @param postProcessorPtr  TODO
         * @return                  TODO
         */
        AlgorithmBuilder& setPostProcessor(std::unique_ptr<IPostProcessor> postProcessorPtr);

        /**
         * TODO
         *
         * @param stoppingCriterionPtr  TODO
         * @return                      TODO
         */
        AlgorithmBuilder& addStoppingCriterion(std::unique_ptr<IStoppingCriterion> stoppingCriterionPtr);

        /**
         * TODO
         *
         * @return TODO
         */
        std::unique_ptr<IRuleModelAssemblage> build() const;

};
