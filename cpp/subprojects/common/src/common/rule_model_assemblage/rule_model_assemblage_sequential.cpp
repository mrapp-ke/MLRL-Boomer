#include "common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"

/**
 * Allows to sequentially induce several rules, optionally starting with a default rule, that are added to a rule-based
 * model.
 */
class SequentialRuleModelAssemblage final : public IRuleModelAssemblage {
    private:

        const std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr_;

        const std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr_;

        const std::unique_ptr<IRulePruningFactory> rulePruningFactoryPtr_;

        const std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr_;

        const std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr_;

        const bool useDefaultRule_;

    public:

        /**
         * @param instanceSamplingFactoryPtr    An unique pointer to an object of type `IInstanceSamplingFactory` that
         *                                      allows create the implementation to be used for sampling the examples
         *                                      whenever a new rule is induced
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
         * @param useDefaultRule                True, if a default rule should be used, False otherwise
         */
        SequentialRuleModelAssemblage(std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
                                      std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
                                      std::unique_ptr<IRulePruningFactory> rulePruningFactoryPtr,
                                      std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
                                      std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr,
                                      bool useDefaultRule)
            : instanceSamplingFactoryPtr_(std::move(instanceSamplingFactoryPtr)),
              featureSamplingFactoryPtr_(std::move(featureSamplingFactoryPtr)),
              rulePruningFactoryPtr_(std::move(rulePruningFactoryPtr)),
              postProcessorFactoryPtr_(std::move(postProcessorFactoryPtr)),
              stoppingCriterionFactoryPtr_(std::move(stoppingCriterionFactoryPtr)), useDefaultRule_(useDefaultRule) {}

        void induceRules(const IFeatureInfo& featureInfo, const IColumnWiseFeatureMatrix& featureMatrix,
                         const IRowWiseLabelMatrix& labelMatrix, const IRuleInduction& ruleInduction,
                         IPartition& partition, ILabelSampling& labelSampling, IStatisticsProvider& statisticsProvider,
                         IThresholds& thresholds, IModelBuilder& modelBuilder, RNG& rng) const override {
            uint32 numRules = useDefaultRule_ ? 1 : 0;
            uint32 numUsedRules = 0;

            // Induce default rule, if necessary...
            if (useDefaultRule_) {
                ruleInduction.induceDefaultRule(statisticsProvider.get(), modelBuilder);
            }

            statisticsProvider.switchToRegularRuleEvaluation();

            // Induce the remaining rules...
            std::unique_ptr<IInstanceSampling> instanceSamplingPtr =
              partition.createInstanceSampling(*instanceSamplingFactoryPtr_, labelMatrix, statisticsProvider.get());
            std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr_->create();
            std::unique_ptr<IRulePruning> rulePruningPtr = rulePruningFactoryPtr_->create();
            std::unique_ptr<IPostProcessor> postProcessorPtr = postProcessorFactoryPtr_->create();
            std::unique_ptr<IStoppingCriterion> stoppingCriterionPtr =
              partition.createStoppingCriterion(*stoppingCriterionFactoryPtr_);

            while (true) {
                IStoppingCriterion::Result stoppingCriterionResult =
                  stoppingCriterionPtr->test(statisticsProvider.get(), numRules);

                if (stoppingCriterionResult.numUsedRules != 0) {
                    numUsedRules = stoppingCriterionResult.numUsedRules;
                }

                if (stoppingCriterionResult.stop) {
                    break;
                }

                const IWeightVector& weights = instanceSamplingPtr->sample(rng);
                const IIndexVector& labelIndices = labelSampling.sample(rng);
                bool success =
                  ruleInduction.induceRule(thresholds, labelIndices, weights, partition, *featureSamplingPtr,
                                           *rulePruningPtr, *postProcessorPtr, rng, modelBuilder);

                if (success) {
                    numRules++;
                } else {
                    break;
                }
            }

            // Set the number of used rules...
            modelBuilder.setNumUsedRules(numUsedRules);
        }
};

/**
 * A factory that allows to create instances of the class `IRuleModelAssemblage` that allow to sequentially induce
 * several rules, optionally starting with a default rule, that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageFactory final : public IRuleModelAssemblageFactory {
    private:

        const bool useDefaultRule_;

    public:

        /**
         * @param useDefaultRule True, if a default rule should be used, false otherwise
         */
        SequentialRuleModelAssemblageFactory(bool useDefaultRule) : useDefaultRule_(useDefaultRule) {}

        std::unique_ptr<IRuleModelAssemblage> create(
          std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
          std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
          std::unique_ptr<IRulePruningFactory> rulePruningFactoryPtr,
          std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
          std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) const override {
            return std::make_unique<SequentialRuleModelAssemblage>(
              std::move(instanceSamplingFactoryPtr), std::move(featureSamplingFactoryPtr),
              std::move(rulePruningFactoryPtr), std::move(postProcessorFactoryPtr),
              std::move(stoppingCriterionFactoryPtr), useDefaultRule_);
        }
};

SequentialRuleModelAssemblageConfig::SequentialRuleModelAssemblageConfig(
  const std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr)
    : defaultRuleConfigPtr_(defaultRuleConfigPtr) {}

std::unique_ptr<IRuleModelAssemblageFactory> SequentialRuleModelAssemblageConfig::createRuleModelAssemblageFactory(
  const IRowWiseLabelMatrix& labelMatrix) const {
    bool useDefaultRule = defaultRuleConfigPtr_->isDefaultRuleUsed(labelMatrix);
    return std::make_unique<SequentialRuleModelAssemblageFactory>(useDefaultRule);
}
