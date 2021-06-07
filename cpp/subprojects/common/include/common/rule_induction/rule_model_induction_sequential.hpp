/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_induction/rule_model_induction.hpp"
#include "common/rule_induction/rule_induction.hpp"
#include "common/sampling/label_sampling.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/feature_sampling.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/statistics/statistics_provider_factory.hpp"
#include "common/stopping/stopping_criterion.hpp"
#include "common/thresholds/thresholds_factory.hpp"
//#include "seco/sampling/instance_sampling_random.hpp"
#include <forward_list>


/**
 * Allows to sequentially induce several rules, starting with a default rule, that will be added to a resulting
 * `RuleModel`.
 */
class SequentialRuleModelInduction : public IRuleModelInduction {

    private:

        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr_;

        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr_;

        std::shared_ptr<IRuleInduction> ruleInductionPtr_;

        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr_;

        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr_;

        std::shared_ptr<ILabelSubSamplingFactory> labelSubSamplingFactoryPtr_;

        std::shared_ptr<IInstanceSubSamplingFactory> instanceSubSamplingFactoryPtr_;

        std::shared_ptr<IFeatureSubSamplingFactory> featureSubSamplingFactoryPtr_;

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
         * @param labelSubSamplingFactoryPtr            A shared pointer to an object of type `ILabelSubSamplingFactory`
         *                                              that allows to create the implementation to be used for sampling
         *                                              the labels whenever a new rule is induced
         * @param instanceSubSamplingFactoryPtr         A shared pointer to an object of type
         *                                              `IInstanceSubSamplingFactory` that allows create the
         *                                              implementation to be used for sampling the examples whenever a
         *                                              new rule is induced
         * @param featureSubSamplingFactoryPtr          A shared pointer to an object of type
         *                                              `IFeatureSubSamplingFactory` that allows to create the
         *                                              implementation to be used for sampling the features that may be
         *                                              used by the conditions of a rule
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
        SequentialRuleModelInduction(
            std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
            std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
            std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
            std::shared_ptr<ILabelSubSamplingFactory> labelSubSamplingFactoryPtr,
            std::shared_ptr<IInstanceSubSamplingFactory> instanceSubSamplingFactoryPtr,
            std::shared_ptr<IFeatureSubSamplingFactory> featureSubSamplingFactoryPtr,
            std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::shared_ptr<IPruning> pruningPtr, std::shared_ptr<IPostProcessor> postProcessorPtr,
            std::unique_ptr<std::forward_list<std::shared_ptr<IStoppingCriterion>>> stoppingCriteriaPtr);

        std::unique_ptr<RuleModel> induceRules(std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                                               std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                               std::shared_ptr<ILabelMatrix> labelMatrixPtr, RNG& rng,
                                               IModelBuilder& modelBuilder) override;

};
