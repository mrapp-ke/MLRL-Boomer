#include "mlrl/common/post_optimization/post_optimization_sequential.hpp"

#include "mlrl/common/sampling/feature_sampling_predefined.hpp"
#include "mlrl/common/util/validation.hpp"

#include <unordered_set>

/**
 * An implementation of the class `IModelBuilder` that allows to replace a single rule of an `IntermediateModelBuilder`.
 */
class RuleReplacementBuilder final : public IModelBuilder {
    private:

        IntermediateModelBuilder::IntermediateRule& intermediateRule_;

    public:

        /**
         * @param intermediateRule A reference to an object of type `IntermediateModelBuilder::IntermediateRule` that
         *                         should be replaced
         */
        RuleReplacementBuilder(IntermediateModelBuilder::IntermediateRule& intermediateRule)
            : intermediateRule_(intermediateRule) {}

        void setDefaultRule(std::unique_ptr<IEvaluatedPrediction>& predictionPtr) override {}

        void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                     std::unique_ptr<IEvaluatedPrediction>& predictionPtr) override {
            intermediateRule_.first = std::move(conditionListPtr);
            intermediateRule_.second = std::move(predictionPtr);
        }

        void setNumUsedRules(uint32 numUsedRules) override {}

        std::unique_ptr<IRuleModel> buildModel() override {
            return nullptr;
        }
};

/**
 * An implementation of the class `IPostOptimizationPhase` that optimizes each rule in a model by relearning it in the
 * context of the other rules.
 */
class SequentialPostOptimization final : public IPostOptimizationPhase {
    private:

        const std::unique_ptr<IRuleInduction> ruleInductionPtr_;

        IntermediateModelBuilder& modelBuilder_;

        const uint32 numIterations_;

        const bool refineHeads_;

        const bool resampleFeatures_;

    public:

        /**
         * @param ruleInductionPtr  An unique pointer to an object of type `IRuleInduction` that should be used for
         *                          inducing new rules
         * @param modelBuilder      A reference to an object of type `IntermediateModelBuilder` that provides access to
         *                          the existing rules
         * @param numIterations     The number of iterations to be performed. Must be at least 1
         * @param refineHeads       True, if the heads of rules should be refined when being relearned, false otherwise
         * @param resampleFeatures  True, if a new sample of the available features should be created when refining a
         *                          new rule, false otherwise
         */
        SequentialPostOptimization(std::unique_ptr<IRuleInduction> ruleInductionPtr,
                                   IntermediateModelBuilder& modelBuilder, uint32 numIterations, bool refineHeads,
                                   bool resampleFeatures)
            : ruleInductionPtr_(std::move(ruleInductionPtr)), modelBuilder_(modelBuilder),
              numIterations_(numIterations), refineHeads_(refineHeads), resampleFeatures_(resampleFeatures) {}

        void optimizeModel(IPartition& partition, IOutputSampling& outputSampling, IInstanceSampling& instanceSampling,
                           IFeatureSampling& featureSampling, IFeatureSpace& featureSpace) const override {
            for (uint32 i = 0; i < numIterations_; i++) {
                for (auto it = modelBuilder_.begin(); it != modelBuilder_.end(); it++) {
                    IntermediateModelBuilder::IntermediateRule& intermediateRule = *it;
                    const ConditionList& conditionList = *intermediateRule.first;
                    IEvaluatedPrediction& prediction = *intermediateRule.second;

                    // Create a new subset of the given thresholds...
                    const IWeightVector& weights = instanceSampling.sample();
                    std::unique_ptr<IFeatureSubspace> featureSubspacePtr = weights.createFeatureSubspace(featureSpace);

                    // Filter the thresholds subset according to the conditions of the current rule...
                    for (auto it2 = conditionList.cbegin(); it2 != conditionList.cend(); it2++) {
                        const Condition& condition = *it2;
                        featureSubspacePtr->filterSubspace(condition);
                    }

                    // Revert the statistics based on the predictions of the current rule...
                    featureSubspacePtr->revertPrediction(prediction);

                    // Learn a new rule...
                    const IIndexVector& outputIndices = refineHeads_ ? outputSampling.sample() : prediction;
                    RuleReplacementBuilder ruleReplacementBuilder(intermediateRule);

                    if (resampleFeatures_) {
                        ruleInductionPtr_->induceRule(featureSpace, outputIndices, weights, partition, featureSampling,
                                                      ruleReplacementBuilder);
                    } else {
                        std::unordered_set<uint32> uniqueFeatureIndices;

                        for (auto it2 = conditionList.cbegin(); it2 != conditionList.cend(); it2++) {
                            const Condition& condition = *it2;
                            uniqueFeatureIndices.emplace(condition.featureIndex);
                        }

                        PartialIndexVector indexVector(uniqueFeatureIndices.size());
                        PartialIndexVector::iterator indexIterator = indexVector.begin();

                        for (auto it2 = uniqueFeatureIndices.cbegin(); it2 != uniqueFeatureIndices.cend(); it2++) {
                            *indexIterator = *it2;
                            indexIterator++;
                        }

                        PredefinedFeatureSampling predefinedFeatureSampling(indexVector);
                        ruleInductionPtr_->induceRule(featureSpace, outputIndices, weights, partition,
                                                      predefinedFeatureSampling, ruleReplacementBuilder);
                    }
                }
            }
        }
};

/**
 * Allows to create instances of the type `IPostOptimizationPhase` that optimize each rule in a model by relearning it
 * in the context of the other rules.
 */
class SequentialPostOptimizationFactory final : public IPostOptimizationPhaseFactory {
    private:

        const std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr_;

        const uint32 numIterations_;

        const bool refineHeads_;

        const bool resampleFeatures_;

    public:

        /**
         * @param ruleInductionFactoryPtr   An unique pointer to an object of type `IRuleInductionFactory`
         * @param numIterations             The number of iterations to be performed. Must be at least 1
         * @param refineHeads               True, if the heads of rules should be refined when being relearned, false
         *                                  otherwise
         * @param resampleFeatures          True, if a new sample of the available features should be created when
         *                                  refining a new rule, false otherwise
         */
        SequentialPostOptimizationFactory(std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
                                          uint32 numIterations, bool refineHeads, bool resampleFeatures)
            : ruleInductionFactoryPtr_(std::move(ruleInductionFactoryPtr)), numIterations_(numIterations),
              refineHeads_(refineHeads), resampleFeatures_(resampleFeatures) {}

        std::unique_ptr<IPostOptimizationPhase> create(IntermediateModelBuilder& modelBuilder) const override {
            return std::make_unique<SequentialPostOptimization>(ruleInductionFactoryPtr_->create(), modelBuilder,
                                                                numIterations_, refineHeads_, resampleFeatures_);
        }
};

SequentialPostOptimizationConfig::SequentialPostOptimizationConfig(
  ReadableProperty<IRuleInductionConfig> ruleInductionConfig)
    : ruleInductionConfig_(ruleInductionConfig), numIterations_(2), refineHeads_(false), resampleFeatures_(true) {}

uint32 SequentialPostOptimizationConfig::getNumIterations() const {
    return numIterations_;
}

ISequentialPostOptimizationConfig& SequentialPostOptimizationConfig::setNumIterations(uint32 numIterations) {
    util::assertGreaterOrEqual<uint32>("numIterations", numIterations, 1);
    numIterations_ = numIterations;
    return *this;
}

bool SequentialPostOptimizationConfig::areHeadsRefined() const {
    return refineHeads_;
}

ISequentialPostOptimizationConfig& SequentialPostOptimizationConfig::setRefineHeads(bool refineHeads) {
    refineHeads_ = refineHeads;
    return *this;
}

bool SequentialPostOptimizationConfig::areFeaturesResampled() const {
    return resampleFeatures_;
}

ISequentialPostOptimizationConfig& SequentialPostOptimizationConfig::setResampleFeatures(bool resampleFeatures) {
    resampleFeatures_ = resampleFeatures;
    return *this;
}

std::unique_ptr<IPostOptimizationPhaseFactory> SequentialPostOptimizationConfig::createPostOptimizationPhaseFactory(
  const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
    return std::make_unique<SequentialPostOptimizationFactory>(
      ruleInductionConfig_.get().createRuleInductionFactory(featureMatrix, outputMatrix), numIterations_, refineHeads_,
      resampleFeatures_);
}
