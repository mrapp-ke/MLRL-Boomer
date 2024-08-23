/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/rule_induction/rule_induction.hpp"
#include "mlrl/common/rule_refinement/score_processor.hpp"

#include <memory>
#include <utility>

/**
 * An abstract base class for all classes that implement an algorithm for the induction of individual rules.
 */
class AbstractRuleInduction : public IRuleInduction {
    private:

        const std::unique_ptr<IRulePruning> rulePruningPtr_;

        const std::unique_ptr<IPostProcessor> postProcessorPtr_;

        const bool recalculatePredictions_;

    protected:

        /**
         * Must be implemented by subclasses in order to grow a rule.
         *
         * @param featureSpace      A reference to an object of type `IFeatureSpace` that provides access to the feature
         *                          space
         * @param outputIndices     A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the outputs for which the rule may predict
         * @param weights           A reference to an object of type `IWeightVector` that provides access to the weights
         *                          of individual training examples
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that belong to the training set and the holdout set,
         *                          respectively
         * @param featureSampling   A reference to an object of type `IFeatureSampling` that should be used for sampling
         *                          the features that may be used by a new condition
         * @param conditionListPtr  A reference to an unique pointer of type `ConditionList` that should be used to
         *                          store the conditions of the rule
         * @param headPtr           A reference to an unique pointer of type `IEvaluatedPrediction` that should be used
         *                          to store the head of the rule
         * @return                  An unique pointer to an object of type `IFeatureSubspace` that has been used to
         *                          grow the rule
         */
        virtual std::unique_ptr<IFeatureSubspace> growRule(IFeatureSpace& featureSpace,
                                                           const IIndexVector& outputIndices,
                                                           const IWeightVector& weights, IPartition& partition,
                                                           IFeatureSampling& featureSampling,
                                                           std::unique_ptr<ConditionList>& conditionListPtr,
                                                           std::unique_ptr<IEvaluatedPrediction>& headPtr) const = 0;

    public:

        /**
         * @param rulePruningPtr         An unique pointer to an object of type `IRulePruning` to be used for pruning
         *                               rules
         * @param postProcessorPtr       An unique pointer to an object of type `IPostProcessor` to be used for
         *                               post-processing the predictions of rules
         * @param recalculatePredictions True, if the predictions of rules should be recalculated on all training
         *                               examples, if some of the examples have zero weights, false otherwise
         */
        explicit AbstractRuleInduction(std::unique_ptr<IRulePruning> rulePruningPtr,
                                       std::unique_ptr<IPostProcessor> postProcessorPtr, bool recalculatePredictions)
            : rulePruningPtr_(std::move(rulePruningPtr)), postProcessorPtr_(std::move(postProcessorPtr)),
              recalculatePredictions_(recalculatePredictions) {}

        virtual ~AbstractRuleInduction() override {}

        void induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const override final {
            uint32 numStatistics = statistics.getNumStatistics();
            uint32 numOutputs = statistics.getNumOutputs();
            CompleteIndexVector outputIndices(numOutputs);
            EqualWeightVector weights(numStatistics);
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(outputIndices, weights);

            for (uint32 i = 0; i < numStatistics; i++) {
                statisticsSubsetPtr->addToSubset(i);
            }

            const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();
            std::unique_ptr<IEvaluatedPrediction> defaultPredictionPtr;
            ScoreProcessor scoreProcessor(defaultPredictionPtr);
            scoreProcessor.processScores(scoreVector);

            std::unique_ptr<IStatisticsUpdate> statisticsUpdatePtr =
              defaultPredictionPtr->createStatisticsUpdate(statistics);

            for (uint32 i = 0; i < numStatistics; i++) {
                statisticsUpdatePtr->applyPrediction(i);
            }

            modelBuilder.setDefaultRule(defaultPredictionPtr);
        }

        bool induceRule(IFeatureSpace& featureSpace, const IIndexVector& outputIndices, const IWeightVector& weights,
                        IPartition& partition, IFeatureSampling& featureSampling,
                        IModelBuilder& modelBuilder) const override final {
            std::unique_ptr<ConditionList> conditionListPtr;
            std::unique_ptr<IEvaluatedPrediction> headPtr;
            std::unique_ptr<IFeatureSubspace> featureSubspacePtr = this->growRule(
              featureSpace, outputIndices, weights, partition, featureSampling, conditionListPtr, headPtr);

            if (headPtr) {
                if (weights.hasZeroWeights()) {
                    // Prune rule...
                    IStatisticsProvider& statisticsProvider = featureSpace.getStatisticsProvider();
                    statisticsProvider.switchToPruningRuleEvaluation();
                    std::unique_ptr<CoverageMask> coverageMaskPtr =
                      rulePruningPtr_->prune(*featureSubspacePtr, partition, *conditionListPtr, *headPtr);
                    statisticsProvider.switchToRegularRuleEvaluation();

                    // Re-calculate the scores in the head based on the entire training data...
                    if (recalculatePredictions_) {
                        const CoverageMask& coverageMask =
                          coverageMaskPtr ? *coverageMaskPtr : featureSubspacePtr->getCoverageMask();
                        partition.recalculatePrediction(*featureSubspacePtr, coverageMask, *headPtr);
                    }
                }

                // Apply post-processor...
                headPtr->postProcess(*postProcessorPtr_);

                // Update the statistics by applying the predictions of the new rule...
                featureSubspacePtr->applyPrediction(*headPtr);

                // Add the induced rule to the model...
                modelBuilder.addRule(conditionListPtr, headPtr);
                return true;
            } else {
                // No rule could be induced, because no useful condition could be found. This might be the case, if all
                // examples have the same values for the considered features.
                return false;
            }
        }
};
