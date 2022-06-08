#include "common/rule_induction/rule_induction_top_down_beam_search.hpp"
#include "common/util/validation.hpp"
#include "rule_induction_common.hpp"


/**
 * An implementation of the type `IRuleInduction` that allows to induce classification rules by using a top-down beam
 * search.
 */
class BeamSearchTopDownRuleInduction final : public AbstractRuleInduction {

    private:

        uint32 beamWidth_;

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        uint32 numThreads_;

    public:

        /**
         * @param beamWidth                 The width that should be used by the beam search. Must be at least 2
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 2 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refinement after a
         *                                  new condition has been added to its body. Must be at least 1 or 0, if the
         *                                  number of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        BeamSearchTopDownRuleInduction(uint32 beamWidth, uint32 minCoverage, uint32 maxConditions,
                                       uint32 maxHeadRefinements, bool recalculatePredictions, uint32 numThreads)
            : AbstractRuleInduction(recalculatePredictions),
              beamWidth_(beamWidth), minCoverage_(minCoverage), maxConditions_(maxConditions),
              maxHeadRefinements_(maxHeadRefinements), numThreads_(numThreads) {

        }

    protected:

        std::unique_ptr<IThresholdsSubset> growRule(
                IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                IPartition& partition, IFeatureSampling& featureSampling, RNG& rng,
                std::unique_ptr<ConditionList>& conditionListPtr,
                std::unique_ptr<AbstractEvaluatedPrediction>& headPtr) const override {
            // TODO Implement
            return nullptr;
        }

};

/**
 * Allows to create instances of the type `IRuleInduction` that induce classification rules by using a top-down beam
 * search, where new conditions are added iteratively to the (initially empty) body of a rule. At each iteration, the
 * refinement that improves the rule the most is chosen. The search stops if no refinement results in an improvement.
 */
class BeamSearchTopDownRuleInductionFactory final : public IRuleInductionFactory {

    private:

        uint32 beamWidth_;

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        bool recalculatePredictions_;

        uint32 numThreads_;

    public:

        /**
         * @param beamWidth                 The width that should be used by the beam search. Must be at least 2
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 2 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refined after a new
         *                                  condition has been added to its body. Must be at least 1 or 0, if the number
         *                                  of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        BeamSearchTopDownRuleInductionFactory(uint32 beamWidth, uint32 minCoverage, uint32 maxConditions,
                                              uint32 maxHeadRefinements, bool recalculatePredictions, uint32 numThreads)
            : beamWidth_(beamWidth), minCoverage_(minCoverage), maxConditions_(maxConditions),
              maxHeadRefinements_(maxHeadRefinements), recalculatePredictions_(recalculatePredictions),
              numThreads_(numThreads) {

        }

        std::unique_ptr<IRuleInduction> create() const override {
            return std::make_unique<BeamSearchTopDownRuleInduction>(beamWidth_, minCoverage_, maxConditions_,
                                                                    maxHeadRefinements_, recalculatePredictions_,
                                                                    numThreads_);
        }

};


BeamSearchTopDownRuleInductionConfig::BeamSearchTopDownRuleInductionConfig(
        const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : beamWidth_(2), minCoverage_(1), maxConditions_(0), maxHeadRefinements_(1), recalculatePredictions_(true),
      multiThreadingConfigPtr_(multiThreadingConfigPtr) {

}

uint32 BeamSearchTopDownRuleInductionConfig::getBeamWidth() const {
    return beamWidth_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setBeamWidth(uint32 beamWidth) {
    assertGreaterOrEqual<uint32>("beamWidth", beamWidth, 2);
    beamWidth_ = beamWidth;
    return *this;
}

uint32 BeamSearchTopDownRuleInductionConfig::getMinCoverage() const {
    return minCoverage_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setMinCoverage(uint32 minCoverage) {
    assertGreaterOrEqual<uint32>("minCoverage", minCoverage, 1);
    minCoverage_ = minCoverage;
    return *this;
}

uint32 BeamSearchTopDownRuleInductionConfig::getMaxConditions() const {
    return maxConditions_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setMaxConditions(uint32 maxConditions) {
    if (maxConditions != 0) { assertGreaterOrEqual<uint32>("maxConditions", maxConditions, 2); }
    maxConditions_ = maxConditions;
    return *this;
}

uint32 BeamSearchTopDownRuleInductionConfig::getMaxHeadRefinements() const {
    return maxHeadRefinements_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setMaxHeadRefinements(
        uint32 maxHeadRefinements) {
    if (maxHeadRefinements != 0) { assertGreaterOrEqual<uint32>("maxHeadRefinements", maxHeadRefinements, 1); }
    maxHeadRefinements_ = maxHeadRefinements;
    return *this;
}

bool BeamSearchTopDownRuleInductionConfig::arePredictionsRecalculated() const {
    return recalculatePredictions_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setRecalculatePredictions(
        bool recalculatePredictions) {
    recalculatePredictions_ = recalculatePredictions;
    return *this;
}

std::unique_ptr<IRuleInductionFactory> BeamSearchTopDownRuleInductionConfig::createRuleInductionFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<BeamSearchTopDownRuleInductionFactory>(beamWidth_, minCoverage_, maxConditions_,
                                                                   maxHeadRefinements_, recalculatePredictions_,
                                                                   numThreads);
}
