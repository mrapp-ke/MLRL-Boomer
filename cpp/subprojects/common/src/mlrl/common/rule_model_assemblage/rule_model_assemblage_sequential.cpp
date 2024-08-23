#include "mlrl/common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"

/**
 * Allows to sequentially induce several rules, optionally starting with a default rule, that are added to a rule-based
 * model.
 */
class SequentialRuleModelAssemblage final : public IRuleModelAssemblage {
    private:

        const std::unique_ptr<IRuleInduction> ruleInductionPtr_;

        const std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr_;

        const bool useDefaultRule_;

    public:

        /**
         * @param ruleInductionPtr              An unique pointer to an object of type `IRuleInduction` to be used for
         *                                      the induction of individual rules
         * @param stoppingCriterionFactoryPtr   An unique pointer to an object of type `IStoppingCriterionFactory` that
         *                                      allows to create the implementations to be used to decide whether
         *                                      additional rules should be induced or not
         * @param useDefaultRule                True, if a default rule should be used, False otherwise
         */
        SequentialRuleModelAssemblage(std::unique_ptr<IRuleInduction> ruleInductionPtr,
                                      std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr,
                                      bool useDefaultRule)
            : ruleInductionPtr_(std::move(ruleInductionPtr)),
              stoppingCriterionFactoryPtr_(std::move(stoppingCriterionFactoryPtr)), useDefaultRule_(useDefaultRule) {}

        void induceRules(const IPostProcessor& postProcessor, IPartition& partition, IOutputSampling& outputSampling,
                         IInstanceSampling& instanceSampling, IFeatureSampling& featureSampling,
                         IStatisticsProvider& statisticsProvider, IFeatureSpace& featureSpace,
                         IModelBuilder& modelBuilder) const override {
            uint32 numRules = useDefaultRule_ ? 1 : 0;
            uint32 numUsedRules = 0;

            // Induce default rule, if necessary...
            if (useDefaultRule_) {
                ruleInductionPtr_->induceDefaultRule(statisticsProvider.get(), modelBuilder);
            }

            statisticsProvider.switchToRegularRuleEvaluation();

            // Induce the remaining rules...
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

                const IWeightVector& weights = instanceSampling.sample();
                const IIndexVector& outputIndices = outputSampling.sample();
                bool success = ruleInductionPtr_->induceRule(featureSpace, outputIndices, weights, partition,
                                                             featureSampling, postProcessor, modelBuilder);

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

        const std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr_;

        const bool useDefaultRule_;

    public:

        /**
         * @param ruleInductionFactoryPtr   An unique pointer to an object of type `IRuleInductionFactory`
         * @param useDefaultRule            True, if a default rule should be used, false otherwise
         */
        SequentialRuleModelAssemblageFactory(std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
                                             bool useDefaultRule)
            : ruleInductionFactoryPtr_(std::move(ruleInductionFactoryPtr)), useDefaultRule_(useDefaultRule) {}

        std::unique_ptr<IRuleModelAssemblage> create(
          std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) const override {
            return std::make_unique<SequentialRuleModelAssemblage>(
              ruleInductionFactoryPtr_->create(), std::move(stoppingCriterionFactoryPtr), useDefaultRule_);
        }
};

SequentialRuleModelAssemblageConfig::SequentialRuleModelAssemblageConfig(
  ReadableProperty<IRuleInductionConfig> ruleInductionConfig, ReadableProperty<IDefaultRuleConfig> defaultRuleConfig)
    : ruleInductionConfig_(ruleInductionConfig), defaultRuleConfig_(defaultRuleConfig) {}

std::unique_ptr<IRuleModelAssemblageFactory> SequentialRuleModelAssemblageConfig::createRuleModelAssemblageFactory(
  const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
    bool useDefaultRule = defaultRuleConfig_.get().isDefaultRuleUsed(outputMatrix);
    return std::make_unique<SequentialRuleModelAssemblageFactory>(
      ruleInductionConfig_.get().createRuleInductionFactory(featureMatrix, outputMatrix), useDefaultRule);
}
