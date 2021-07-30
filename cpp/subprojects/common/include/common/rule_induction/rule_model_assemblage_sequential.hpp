/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"


/**
 * Allows to sequentially induce several rules, starting with a default rule, that will be added to a resulting
 * `RuleModel`.
 */
// TODO Move to cpp file
class SequentialRuleModelAssemblage : public IRuleModelAssemblage {

    private:

        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr_;

        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr_;

        std::shared_ptr<IRuleInduction> ruleInductionPtr_;

        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr_;

        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr_;

        std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr_;

        std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr_;

        std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr_;

        std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr_;

        std::shared_ptr<IPruning> pruningPtr_;

        std::shared_ptr<IPostProcessor> postProcessorPtr_;

        std::unique_ptr<std::forward_list<std::shared_ptr<IStoppingCriterion>>> stoppingCriteriaPtr_;

    public:

        /**
         * @param statisticsProviderFactoryPtr          A shared pointer to an object of type
         *                                              `IStatisticsProviderFactory` that provides access to the
         *                                              statistics which serve as the basis for learning rules
         * @param thresholdsFactoryPtr                  A shared pointer to an object of type `IThresholdsFactory` that
         *                                              allows to create objects that provide access to the thresholds
         *                                              that may be used by the conditions of rules
         * @param ruleInductionPtr                      A shared pointer to an object of type `IRuleInduction` that
         *                                              should be used to induce individual rules
         * @param defaultRuleHeadRefinementFactoryPtr   A shared pointer to an object of type `IHeadRefinement` that
         *                                              allows to create instances of the class that should be used to
         *                                              find the head of the default rule
         * @param headRefinementFactoryPtr              A shared pointer to an object of type `IHeadRefinement` that
         *                                              allows to create instances of the class that should be used to
         *                                              find the head of all remaining rules
         * @param labelSamplingFactoryPtr               A shared pointer to an object of type `ILabelSamplingFactory`
         *                                              that allows to create the implementation to be used for sampling
         *                                              the labels whenever a new rule is induced
         * @param instanceSamplingFactoryPtr            A shared pointer to an object of type `IInstanceSamplingFactory`
         *                                              that allows create the implementation to be used for sampling
         *                                              the examples whenever a new rule is induced
         * @param featureSamplingFactoryPtr             A shared pointer to an object of type `IFeatureSamplingFactory`
         *                                              that allows to create the implementation to be used for sampling
         *                                              the features that may be used by the conditions of a rule
         * @param partitionSamplingFactoryPtr           A shared pointer to an object of type
         *                                              `IPartitionSamplingFactory` that allows to create the
         *                                              implementation to be used for partitioning the training examples
         *                                              into a training set and a holdout set
         * @param pruningPtr                            A shared pointer to an object of type `IPruning` that should be
         *                                              used to prune the rules
         * @param postProcessorPtr                      A shared pointer to an object of type `IPostProcessor` that
         *                                              should be used to post-process the predictions of rules
         * @param stoppingCriteriaPtr                   An unique pointer to a list that contains the stopping criteria,
         *                                              which should be used to decide whether additional rules should
         *                                              be induced or not
         */
        SequentialRuleModelAssemblage(
            std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
            std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
            std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
            std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::shared_ptr<IPruning> pruningPtr, std::shared_ptr<IPostProcessor> postProcessorPtr,
            std::unique_ptr<std::forward_list<std::shared_ptr<IStoppingCriterion>>> stoppingCriteriaPtr);

        std::unique_ptr<RuleModel> induceRules(const INominalFeatureMask& nominalFeatureMask,
                                               const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
                                               uint32 randomState, IModelBuilder& modelBuilder) override;

};

/**
 * A factory that allows to create instances of the class `SequentialRuleModelAssemblage`.
 */
class SequentialRuleModelAssemblageFactory final : public IRuleModelAssemblageFactory {

    public:

        std::unique_ptr<IRuleModelAssemblage> create(
            std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
            std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
            std::shared_ptr<IHeadRefinementFactory> regularRuleHeadRefinementFactoryPtr,
            std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::shared_ptr<IPruning> pruningPtr, std::shared_ptr<IPostProcessor> postProcessorPtr,
            const std::forward_list<std::shared_ptr<IStoppingCriterion>> stoppingCriteria) const override;

};
