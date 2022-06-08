#include "common/rule_induction/rule_induction_top_down_greedy.hpp"
#include "common/util/validation.hpp"
#include "rule_induction_common.hpp"
#include "omp.h"


/**
 * Stores an unique pointer to an object of type `IRuleRefinement` that may be used to search for potential refinements
 * of a rule, as well as to an object of type `SingleRefinementComparator` that allows comparing different refinements
 * and keeping track of the best one.
 */
struct RuleRefinement {

    std::unique_ptr<IRuleRefinement> ruleRefinementPtr;

    std::unique_ptr<SingleRefinementComparator> comparatorPtr;

};

/**
 * An implementation of the type `IRuleInduction` that allows to induce classification rules by using a greedy top-down
 * search.
 */
class GreedyTopDownRuleInduction final : public AbstractRuleInduction {

    private:

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        bool recalculatePredictions_;

        uint32 numThreads_;

    public:

        /**
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 1 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refinement after a
         *                                  new condition has been added to its body. Must be at least 1 or 0, if the
         *                                  number of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        GreedyTopDownRuleInduction(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                   bool recalculatePredictions, uint32 numThreads)
            : minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
              recalculatePredictions_(recalculatePredictions), numThreads_(numThreads) {

        }

        bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
                        IPartition& partition, IFeatureSampling& featureSampling, const IPruning& pruning,
                        const IPostProcessor& postProcessor, RNG& rng, IModelBuilder& modelBuilder) const override {
            // The label indices for which the next refinement of the rule may predict
            const IIndexVector* currentLabelIndices = &labelIndices;
            // A list that contains the conditions in the rule's body (in the order they have been learned)
            std::unique_ptr<ConditionList> conditionListPtr = std::make_unique<ConditionList>();
            // The comparator that is used to keep track of the best refinement of the rule
            SingleRefinementComparator refinementComparator;
            // Whether a refinement of the current rule has been found
            bool foundRefinement = true;

            // Create a new subset of the given thresholds...
            std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = weights.createThresholdsSubset(thresholds);

            // Search for the best refinement until no improvement in terms of the rule's quality score is possible
            // anymore or the maximum number of conditions has been reached...
            while (foundRefinement && (maxConditions_ == 0 || conditionListPtr->getNumConditions() < maxConditions_)) {
                foundRefinement = false;

                // Sample features...
                const IIndexVector& sampledFeatureIndices = featureSampling.sample(rng);
                uint32 numSampledFeatures = sampledFeatureIndices.getNumElements();

                // For each feature, create an object of type `IRuleRefinement`...
                RuleRefinement* ruleRefinements = new RuleRefinement[numSampledFeatures];

                for (uint32 i = 0; i < numSampledFeatures; i++) {
                    uint32 featureIndex = sampledFeatureIndices.getIndex(i);
                    RuleRefinement& ruleRefinement = ruleRefinements[i];
                    ruleRefinement.comparatorPtr = std::make_unique<SingleRefinementComparator>(refinementComparator);
                    ruleRefinement.ruleRefinementPtr =
                        currentLabelIndices->createRuleRefinement(*thresholdsSubsetPtr, featureIndex);
                }

                // Search for the best condition among all available features to be added to the current rule...
                #pragma omp parallel for firstprivate(numSampledFeatures) firstprivate(ruleRefinements) \
                schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numSampledFeatures; i++) {
                    RuleRefinement& ruleRefinement = ruleRefinements[i];
                    ruleRefinement.ruleRefinementPtr->findRefinement(*ruleRefinement.comparatorPtr);
                }

                // Pick the best refinement among the refinements that have been found for the different features...
                for (uint32 i = 0; i < numSampledFeatures; i++) {
                    RuleRefinement& ruleRefinement = ruleRefinements[i];
                    foundRefinement |= refinementComparator.merge(*ruleRefinement.comparatorPtr);
                }

                delete[] ruleRefinements;

                if (foundRefinement) {
                    Refinement& bestRefinement = *refinementComparator.begin();

                    // Sort the rule's predictions by the corresponding label indices...
                    bestRefinement.headPtr->sort();

                    // Filter the current subset of thresholds by applying the best refinement that has been found...
                    thresholdsSubsetPtr->filterThresholds(bestRefinement);

                    // Add the new condition...
                    conditionListPtr->addCondition(bestRefinement);

                    // Keep the labels for which the rule predicts, if the head should not be further refined...
                    if (maxHeadRefinements_ > 0 && conditionListPtr->getNumConditions() >= maxHeadRefinements_) {
                        currentLabelIndices = bestRefinement.headPtr.get();
                    }

                    // Abort refinement process if the rule is not allowed to cover less examples...
                    if (bestRefinement.numCovered <= minCoverage_) {
                        break;
                    }
                }
            }


            if (refinementComparator.getNumElements() > 0) {
                Refinement& bestRefinement = *refinementComparator.begin();

                if (weights.hasZeroWeights()) {
                    // Prune rule...
                    IStatisticsProvider& statisticsProvider = thresholds.getStatisticsProvider();
                    statisticsProvider.switchToPruningRuleEvaluation();
                    std::unique_ptr<ICoverageState> coverageStatePtr = pruning.prune(*thresholdsSubsetPtr, partition,
                                                                                     *conditionListPtr,
                                                                                     *bestRefinement.headPtr);
                    statisticsProvider.switchToRegularRuleEvaluation();

                    // Re-calculate the scores in the head based on the entire training data...
                    if (recalculatePredictions_) {
                        const ICoverageState& coverageState =
                            coverageStatePtr ? *coverageStatePtr : thresholdsSubsetPtr->getCoverageState();
                        partition.recalculatePrediction(*thresholdsSubsetPtr, coverageState, *bestRefinement.headPtr);
                    }
                }

                // Apply post-processor...
                postProcessor.postProcess(*bestRefinement.headPtr);

                // Update the statistics by applying the predictions of the new rule...
                thresholdsSubsetPtr->applyPrediction(*bestRefinement.headPtr);

                // Add the induced rule to the model...
                modelBuilder.addRule(conditionListPtr, bestRefinement.headPtr);
                return true;
            } else {
                // No rule could be induced, because no useful condition could be found. This might be the case, if all
                // examples have the same values for the considered features.
                return false;
            }
        }

};

/**
 * Allows to create instances of the type `IRuleInduction` that induce classification rules by using a greedy top-down
 * search, where new conditions are added iteratively to the (initially empty) body of a rule. At each iteration, the
 * refinement that improves the rule the most is chosen. The search stops if no refinement results in an improvement.
 */
class GreedyTopDownRuleInductionFactory final : public IRuleInductionFactory {

    private:

        uint32 minCoverage_;

        uint32 maxConditions_;

        uint32 maxHeadRefinements_;

        bool recalculatePredictions_;

        uint32 numThreads_;

    public:

        /**
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 1 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refined after a new
         *                                  condition has been added to its body. Must be at least 1 or 0, if the number
         *                                  of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        GreedyTopDownRuleInductionFactory(uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                          bool recalculatePredictions, uint32 numThreads)
            : minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
              recalculatePredictions_(recalculatePredictions), numThreads_(numThreads) {

        }

        std::unique_ptr<IRuleInduction> create() const override {
            return std::make_unique<GreedyTopDownRuleInduction>(minCoverage_, maxConditions_, maxHeadRefinements_,
                                                                recalculatePredictions_, numThreads_);
        }

};


GreedyTopDownRuleInductionConfig::GreedyTopDownRuleInductionConfig(
        const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : minCoverage_(1), maxConditions_(0), maxHeadRefinements_(1), recalculatePredictions_(true),
      multiThreadingConfigPtr_(multiThreadingConfigPtr) {

}

uint32 GreedyTopDownRuleInductionConfig::getMinCoverage() const {
    return minCoverage_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setMinCoverage(uint32 minCoverage) {
    assertGreaterOrEqual<uint32>("minCoverage", minCoverage, 1);
    minCoverage_ = minCoverage;
    return *this;
}

uint32 GreedyTopDownRuleInductionConfig::getMaxConditions() const {
    return maxConditions_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setMaxConditions(uint32 maxConditions) {
    if (maxConditions != 0) { assertGreaterOrEqual<uint32>("maxConditions", maxConditions, 1); }
    maxConditions_ = maxConditions;
    return *this;
}

uint32 GreedyTopDownRuleInductionConfig::getMaxHeadRefinements() const {
    return maxHeadRefinements_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setMaxHeadRefinements(uint32 maxHeadRefinements) {
    if (maxHeadRefinements != 0) { assertGreaterOrEqual<uint32>("maxHeadRefinements", maxHeadRefinements, 1); }
    maxHeadRefinements_ = maxHeadRefinements;
    return *this;
}

bool GreedyTopDownRuleInductionConfig::arePredictionsRecalculated() const {
    return recalculatePredictions_;
}

IGreedyTopDownRuleInductionConfig& GreedyTopDownRuleInductionConfig::setRecalculatePredictions(
        bool recalculatePredictions) {
    recalculatePredictions_ = recalculatePredictions;
    return *this;
}

std::unique_ptr<IRuleInductionFactory> GreedyTopDownRuleInductionConfig::createRuleInductionFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<GreedyTopDownRuleInductionFactory>(minCoverage_, maxConditions_, maxHeadRefinements_,
                                                               recalculatePredictions_, numThreads);
}
