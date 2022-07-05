#include "common/post_optimization/post_optimization_sequential.hpp"
#include "common/util/validation.hpp"


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
            : intermediateRule_(intermediateRule) {

        }

        void setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) override {

        }

        void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                     std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) {
            intermediateRule_.first = std::move(conditionListPtr);
            intermediateRule_.second = std::move(predictionPtr);
        }

        std::unique_ptr<IRuleModel> buildModel(uint32 numUsedRules) {
            return nullptr;
        }

};

/**
 * An implementation of the class `IPostOptimizationPhase` that optimizes each rule in a model by relearning it in the
 * context of the other rules.
 */
class SequentialPostOptimization final : public IPostOptimizationPhase {

    private:

        IntermediateModelBuilder& modelBuilder_;

        uint32 numIterations_;

    public:

        /**
         * @param modelBuilderPtr   A reference to an object of type `IntermediateModelBuilder` that provides access to
         *                          the existing rules
         * @param numIterations     The number of iterations to be performed. Must be at least 1
         */
        SequentialPostOptimization(IntermediateModelBuilder& modelBuilder, uint32 numIterations)
            : modelBuilder_(modelBuilder), numIterations_(numIterations) {

        }

        void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                           ILabelSampling& labelSampling, IInstanceSampling& instanceSampling,
                           IFeatureSampling& featureSampling, const IPruning& pruning,
                           const IPostProcessor& postProcessor, RNG& rng) const override {
            for (uint32 i = 0; i < numIterations_; i++) {
                for (auto it = modelBuilder_.begin(); it != modelBuilder_.end(); it++) {
                    IntermediateModelBuilder::IntermediateRule& intermediateRule = *it;
                    std::unique_ptr<ConditionList>& conditionListPtr = intermediateRule.first;
                    std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr = intermediateRule.second;

                    // Create a new subset of the given thresholds...
                    const IWeightVector& weights = instanceSampling.sample(rng);
                    std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = weights.createThresholdsSubset(thresholds);

                    // Filter the thresholds subset according to the conditions of the current rule...
                    for (auto it2 = conditionListPtr->cbegin(); it2 != conditionListPtr->cend(); it2++) {
                        const Condition& condition = *it2;
                        thresholdsSubsetPtr->filterThresholds(condition);
                    }

                    // Revert the statistics based on the predictions of the current rule...
                    thresholdsSubsetPtr->revertPrediction(*predictionPtr);

                    // Learn a new rule...
                    const IIndexVector& labelIndices = *predictionPtr;
                    RuleReplacementBuilder ruleReplacementBuilder(intermediateRule);
                    ruleInduction.induceRule(thresholds, labelIndices, weights, partition, featureSampling, pruning,
                                             postProcessor, rng, ruleReplacementBuilder);
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

        uint32 numIterations_;

    public:

        /**
         * @param numIterations The number of iterations to be performed. Must be at least 1
         */
        SequentialPostOptimizationFactory(uint32 numIterations)
            : numIterations_(numIterations) {

        }

        std::unique_ptr<IPostOptimizationPhase> create(IntermediateModelBuilder& modelBuilder) const override {
            return std::make_unique<SequentialPostOptimization>(modelBuilder, numIterations_);
        }

};

SequentialPostOptimizationConfig::SequentialPostOptimizationConfig()
    : numIterations_(2) {

}

uint32 SequentialPostOptimizationConfig::getNumIterations() const {
    return numIterations_;
}

ISequentialPostOptimizationConfig& SequentialPostOptimizationConfig::setNumIterations(uint32 numIterations) {
    assertGreaterOrEqual<uint32>("numIterations", numIterations, 1);
    numIterations_ = numIterations;
    return *this;
}

std::unique_ptr<IPostOptimizationPhaseFactory> SequentialPostOptimizationConfig::createPostOptimizationPhaseFactory() const {
    return std::make_unique<SequentialPostOptimizationFactory>(numIterations_);
}
