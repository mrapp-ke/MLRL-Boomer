#include "common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"


/**
 * Allows to sequentially induce several rules, optionally starting with a default rule, that are added to a rule-based
 * model.
 */
class SequentialRuleModelAssemblage final : public IRuleModelAssemblage {

    private:

        std::unique_ptr<IModelBuilderFactory> modelBuilderFactoryPtr_;

        std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr_;

        std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr_;

        std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr_;

        std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr_;

        std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr_;

        std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr_;

        std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr_;

        std::unique_ptr<IPruningFactory> pruningFactoryPtr_;

        std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr_;

        std::unique_ptr<IPostOptimizationFactory> postOptimizationFactoryPtr_;

        std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr_;

        bool useDefaultRule_;

    public:

        /**
         * @param modelBuilderFactoryPtr        An unique pointer to an object of type `IModelBuilderFactory` that
         *                                      allows to create the builder to be used for assembling a model
         * @param statisticsProviderFactoryPtr  An unique pointer to an object of type `IStatisticsProviderFactory` that
         *                                      provides access to the statistics which serve as the basis for learning
         *                                      rules
         * @param thresholdsFactoryPtr          An unique pointer to an object of type `IThresholdsFactory` that allows
         *                                      to create objects that provide access to the thresholds that may be used
         *                                      by the conditions of rules
         * @param ruleInductionFactoryPtr       An unique pointer to an object of type `IRuleInductionFactory` that
         *                                      allows to create the implementation to be used for the induction of
         *                                      individual rules
         * @param labelSamplingFactoryPtr       An unique pointer to an object of type `ILabelSamplingFactory` that
         *                                      allows to create the implementation to be used for sampling the labels
         *                                      whenever a new rule is induced
         * @param instanceSamplingFactoryPtr    An unique pointer to an object of type `IInstanceSamplingFactory` that
         *                                      allows create the implementation to be used for sampling the examples
         *                                      whenever a new rule is induced
         * @param featureSamplingFactoryPtr     An unique pointer to an object of type `IFeatureSamplingFactory` that
         *                                      allows to create the implementation to be used for sampling the features
         *                                      that may be used by the conditions of a rule
         * @param partitionSamplingFactoryPtr   An unique pointer to an object of type `IPartitionSamplingFactory` that
         *                                      allows to create the implementation to be used for partitioning the
         *                                      training examples into a training set and a holdout set
         * @param pruningFactoryPtr             An unique pointer to an object of type `IPruningFactory` that allows to
         *                                      create the implementation to be used for pruning rules
         * @param postProcessorFactoryPtr       An unique pointer to an object of type `IPostProcessorFactory` that
         *                                      allows to create the implementation to be used for post-processing the
         *                                      predictions of rules
         * @param postOptimizationFactoryPtr    An unique pointer to an object of type `IPostOptimizationFactory` that
         *                                      allows to create the implementation to be used for optimizing a
         *                                      rule-based model once it has been learned
         * @param stoppingCriterionFactoryPtr   An unique pointer to an object of type `IStoppingCriterionFactory` that
         *                                      allows to create the implementations to be used to decide whether
         *                                      additional rules should be induced or not
         * @param useDefaultRule                True, if a default rule should be used, False otherwise
         */
        SequentialRuleModelAssemblage(
            std::unique_ptr<IModelBuilderFactory> modelBuilderFactoryPtr,
            std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
            std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
            std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::unique_ptr<IPruningFactory> pruningFactoryPtr,
            std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
            std::unique_ptr<IPostOptimizationFactory> postOptimizationFactoryPtr,
            std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr,
            bool useDefaultRule)
            : modelBuilderFactoryPtr_(std::move(modelBuilderFactoryPtr)),
              statisticsProviderFactoryPtr_(std::move(statisticsProviderFactoryPtr)),
              thresholdsFactoryPtr_(std::move(thresholdsFactoryPtr)),
              ruleInductionFactoryPtr_(std::move(ruleInductionFactoryPtr)),
              labelSamplingFactoryPtr_(std::move(labelSamplingFactoryPtr)),
              instanceSamplingFactoryPtr_(std::move(instanceSamplingFactoryPtr)),
              featureSamplingFactoryPtr_(std::move(featureSamplingFactoryPtr)),
              partitionSamplingFactoryPtr_(std::move(partitionSamplingFactoryPtr)),
              pruningFactoryPtr_(std::move(pruningFactoryPtr)),
              postProcessorFactoryPtr_(std::move(postProcessorFactoryPtr)),
              postOptimizationFactoryPtr_(std::move(postOptimizationFactoryPtr)),
              stoppingCriterionFactoryPtr_(std::move(stoppingCriterionFactoryPtr)),
              useDefaultRule_(useDefaultRule) {

        }

        std::unique_ptr<IRuleModel> induceRules(const IFeatureInfo& featureInfo,
                                                const IColumnWiseFeatureMatrix& featureMatrix,
                                                const IRowWiseLabelMatrix& labelMatrix,
                                                uint32 randomState) const override {
            uint32 numRules = useDefaultRule_ ? 1 : 0;
            uint32 numUsedRules = 0;

            // Partition training data...
            std::unique_ptr<IPartitionSampling> partitionSamplingPtr = labelMatrix.createPartitionSampling(
                *partitionSamplingFactoryPtr_);
            RNG rng(randomState);
            IPartition& partition = partitionSamplingPtr->partition(rng);

            // Induce default rule...
            std::unique_ptr<IPostOptimization> postOptimizationPtr = postOptimizationFactoryPtr_->create(
                *modelBuilderFactoryPtr_);
            std::unique_ptr<IStatisticsProvider> statisticsProviderPtr = labelMatrix.createStatisticsProvider(
                *statisticsProviderFactoryPtr_);
            std::unique_ptr<IRuleInduction> ruleInductionPtr = ruleInductionFactoryPtr_->create();
            IModelBuilder& modelBuilder = postOptimizationPtr->getModelBuilder();

            if (useDefaultRule_) {
                ruleInductionPtr->induceDefaultRule(statisticsProviderPtr->get(), modelBuilder);
            }

            statisticsProviderPtr->switchToRegularRuleEvaluation();

            // Induce the remaining rules...
            std::unique_ptr<IThresholds> thresholdsPtr = thresholdsFactoryPtr_->create(featureMatrix, featureInfo,
                                                                                       *statisticsProviderPtr);
            std::unique_ptr<IInstanceSampling> instanceSamplingPtr = partition.createInstanceSampling(
                *instanceSamplingFactoryPtr_, labelMatrix, statisticsProviderPtr->get());
            std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr_->create();
            std::unique_ptr<ILabelSampling> labelSamplingPtr = labelSamplingFactoryPtr_->create();
            std::unique_ptr<IPruning> pruningPtr = pruningFactoryPtr_->create();
            std::unique_ptr<IPostProcessor> postProcessorPtr = postProcessorFactoryPtr_->create();
            std::unique_ptr<IStoppingCriterion> stoppingCriterionPtr =
                partition.createStoppingCriterion(*stoppingCriterionFactoryPtr_);
            IStoppingCriterion::Result stoppingCriterionResult;

            while (stoppingCriterionResult = stoppingCriterionPtr->test(statisticsProviderPtr->get(), numRules),
                   stoppingCriterionResult.action != IStoppingCriterion::Action::FORCE_STOP) {
                if (stoppingCriterionResult.action == IStoppingCriterion::Action::STORE_STOP && numUsedRules == 0) {
                    numUsedRules = stoppingCriterionResult.numRules;
                }

                const IWeightVector& weights = instanceSamplingPtr->sample(rng);
                const IIndexVector& labelIndices = labelSamplingPtr->sample(rng);
                bool success = ruleInductionPtr->induceRule(*thresholdsPtr, labelIndices, weights, partition,
                                                            *featureSamplingPtr, *pruningPtr, *postProcessorPtr, rng,
                                                            modelBuilder);

                if (success) {
                    numRules++;
                } else {
                    break;
                }
            }

            // Post-optimize the model...
            postOptimizationPtr->optimizeModel(*thresholdsPtr, *ruleInductionPtr, partition, *labelSamplingPtr,
                                               *instanceSamplingPtr, *featureSamplingPtr, *pruningPtr,
                                               *postProcessorPtr, rng);

            // Build and return the final model...
            return modelBuilder.buildModel(numUsedRules);
        }

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
        SequentialRuleModelAssemblageFactory(bool useDefaultRule)
            : useDefaultRule_(useDefaultRule) {

        }

        std::unique_ptr<IRuleModelAssemblage> create(
                std::unique_ptr<IModelBuilderFactory> modelBuilderFactoryPtr,
                std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
                std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
                std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
                std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
                std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
                std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
                std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
                std::unique_ptr<IPruningFactory> pruningFactoryPtr,
                std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
                std::unique_ptr<IPostOptimizationFactory> postOptimizationFactoryPtr,
                std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) const override {
            return std::make_unique<SequentialRuleModelAssemblage>(std::move(modelBuilderFactoryPtr),
                                                                   std::move(statisticsProviderFactoryPtr),
                                                                   std::move(thresholdsFactoryPtr),
                                                                   std::move(ruleInductionFactoryPtr),
                                                                   std::move(labelSamplingFactoryPtr),
                                                                   std::move(instanceSamplingFactoryPtr),
                                                                   std::move(featureSamplingFactoryPtr),
                                                                   std::move(partitionSamplingFactoryPtr),
                                                                   std::move(pruningFactoryPtr),
                                                                   std::move(postProcessorFactoryPtr),
                                                                   std::move(postOptimizationFactoryPtr),
                                                                   std::move(stoppingCriterionFactoryPtr),
                                                                   useDefaultRule_);
        }

};

SequentialRuleModelAssemblageConfig::SequentialRuleModelAssemblageConfig(
        const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr)
    : defaultRuleConfigPtr_(defaultRuleConfigPtr) {

}

std::unique_ptr<IRuleModelAssemblageFactory> SequentialRuleModelAssemblageConfig::createRuleModelAssemblageFactory(
        const IRowWiseLabelMatrix& labelMatrix) const {
    bool useDefaultRule = defaultRuleConfigPtr_->isDefaultRuleUsed(labelMatrix);
    return std::make_unique<SequentialRuleModelAssemblageFactory>(useDefaultRule);
}
