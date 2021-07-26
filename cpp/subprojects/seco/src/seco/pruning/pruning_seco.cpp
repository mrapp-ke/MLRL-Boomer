#include "seco/pruning/pruning_seco.hpp"
#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include "common/rule_evaluation/score_vector_label_wise_binned_dense.hpp"
#include "common/head_refinement/prediction.hpp"
#include "common/debugging/debug.hpp"

#include <iostream>


namespace seco {

    template<typename Iterator>
    static inline std::unique_ptr<SparseArrayVector<float64>> argSort(Iterator iterator, uint32 numElements) {
        std::unique_ptr<SparseArrayVector<float64>> sortedVectorPtr = std::make_unique<SparseArrayVector<float64>>(
                numElements);
        SparseArrayVector<float64>::iterator sortedIterator = sortedVectorPtr->begin();

        for (uint32 i = 0; i < numElements; i++) {
            sortedIterator[i].index = i;
            sortedIterator[i].value = iterator[i];
        }

        sortedVectorPtr->sortByValues();
        return sortedVectorPtr;
    }

    static inline float64 calculateOverallQualityScore(float64 sumOfQualityScores, uint32 numPredictions,
                                                       const ILiftFunction &liftFunction) {
        return 1 - ((sumOfQualityScores / numPredictions) * liftFunction.calculateLift(numPredictions));
    }

    template<typename ScoreVector>
    const AbstractEvaluatedPrediction *SecoPruning::findHeadInternally(const AbstractEvaluatedPrediction *bestHead,
                                                                       const ScoreVector &scoreVector) {
        uint32 numPredictions = scoreVector.getNumElements();
        typename ScoreVector::quality_score_const_iterator qualityScoreIterator = scoreVector.quality_scores_cbegin();
        std::unique_ptr<SparseArrayVector<float64>> sortedVectorPtr;
        uint32 bestNumPredictions;
        float64 bestOverallQualityScore;

        sortedVectorPtr = argSort(qualityScoreIterator, numPredictions);
        SparseArrayVector<float64>::const_iterator sortedIterator = sortedVectorPtr->cbegin();
        float64 maximumLift = liftFunctionPtr_->getMaxLift();
        float64 sumOfQualityScores = 1 - qualityScoreIterator[sortedIterator[0].index];
        bestOverallQualityScore = calculateOverallQualityScore(sumOfQualityScores, 1, *liftFunctionPtr_);
        bestNumPredictions = 1;

        for (uint32 i = 1; i < numPredictions; i++) {
            sumOfQualityScores += 1 - qualityScoreIterator[sortedIterator[i].index];
            uint32 currentNumPredictions = i + 1;
            float64 overallQualityScore = calculateOverallQualityScore(sumOfQualityScores,
                                                                       currentNumPredictions,
                                                                       *liftFunctionPtr_);

            if (overallQualityScore < bestOverallQualityScore) {
                bestNumPredictions = currentNumPredictions;
                bestOverallQualityScore = overallQualityScore;
            }

            if (overallQualityScore * maximumLift < bestOverallQualityScore) {
                // Prunable by decomposition...
                break;
            }
        }

        if (bestHead == nullptr || bestOverallQualityScore < bestHead->overallQualityScore) {
            if (headPtr_ == nullptr) {
                headPtr_ = std::make_unique<PartialPrediction>(bestNumPredictions);
            } else if (headPtr_->getNumElements() != bestNumPredictions) {
                headPtr_->setNumElements(bestNumPredictions, false);
            }

            typename ScoreVector::score_const_iterator scoreIterator = scoreVector.scores_cbegin();
            typename ScoreVector::index_const_iterator indexIterator = scoreVector.indices_cbegin();
            PartialPrediction::score_iterator headScoreIterator = headPtr_->scores_begin();
            PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();
            float64 worstQualityScore = sortedVectorPtr->cbegin()[bestNumPredictions - 1].value;
            uint32 n = 0;

            for (uint32 i = 0; i < numPredictions; i++) {
                if (qualityScoreIterator[i] <= worstQualityScore) {
                    headIndexIterator[n] = indexIterator[i];
                    headScoreIterator[n] = scoreIterator[i];
                    n++;

                    if (n == bestNumPredictions) {
                        break;
                    }
                }
            }

            headPtr_->overallQualityScore = bestOverallQualityScore;
            return headPtr_.get();
        }

        return nullptr;
    }

    const AbstractEvaluatedPrediction *SecoPruning::processScores(
            const AbstractEvaluatedPrediction *bestHead,
            const DenseLabelWiseScoreVector<FullIndexVector> &scoreVector) {
        return findHeadInternally<DenseLabelWiseScoreVector<FullIndexVector>>(bestHead, scoreVector);
    }

    const AbstractEvaluatedPrediction *SecoPruning::processScores(
            const AbstractEvaluatedPrediction *bestHead,
            const DenseLabelWiseScoreVector<PartialIndexVector> &scoreVector) {
        return findHeadInternally<DenseLabelWiseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
    }

    const AbstractEvaluatedPrediction *SecoPruning::processScores(
            const AbstractEvaluatedPrediction *bestHead,
            const DenseBinnedLabelWiseScoreVector<FullIndexVector> &scoreVector) {
        return findHeadInternally<DenseBinnedLabelWiseScoreVector<FullIndexVector>>(bestHead, scoreVector);
    }

    const AbstractEvaluatedPrediction *SecoPruning::processScores(
            const AbstractEvaluatedPrediction *bestHead,
            const DenseBinnedLabelWiseScoreVector<PartialIndexVector> &scoreVector) {
        return findHeadInternally<DenseBinnedLabelWiseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
    }


    SecoPruning::SecoPruning(std::shared_ptr<seco::ILiftFunction> liftFunctionPtr)
            : liftFunctionPtr_(liftFunctionPtr) {

    }

    std::unique_ptr<ICoverageState> SecoPruning::prune(IThresholdsSubset &thresholdsSubset, IPartition &partition,
                                                       ConditionList &conditions,
                                                       const AbstractEvaluatedPrediction *bestHead) const {
        ConditionList::size_type numConditions = conditions.getNumConditions();
        std::unique_ptr<ICoverageState> bestCoverageStatePtr;

        // Only rules with more than one condition can be pruned...
        if (numConditions > 1) {

            Debugger::lb(true);

            // Calculate the quality score of the original rule on the prune set...
            const ICoverageState &originalCoverageState = thresholdsSubset.getCoverageState();
            float64 bestQualityScore = partition.evaluateOutOfSample(thresholdsSubset, originalCoverageState,
                                                                     *bestHead).overallQualityScore;

            // Debugger: print the original coverage mask
            Debugger::printCoverageMask(originalCoverageState, true);

            // Create a copy of the original coverage mask...
            bestCoverageStatePtr = originalCoverageState.copy();

            // Reset the given thresholds...
            thresholdsSubset.resetThresholds();

            // We process the existing rule's conditions (except for the last one) in the order they have been learned. At
            // each iteration, we calculate the quality score of a rule that only contains the conditions processed so far
            // and keep track of the best rule...
            auto conditionIterator = conditions.cbegin();
            ConditionList::size_type numPrunedConditions = 0;

            for (std::list<Condition>::size_type n = 1; n < numConditions; n++) {
                // Filter the thresholds by applying the current condition...
                const Condition &condition = *conditionIterator;
                thresholdsSubset.filterThresholds(condition);


                // Debugger: print rule
                Debugger::printRule(conditions.cbegin(), numConditions, *bestHead);

                // Calculate the quality score of a rule that contains the conditions that have been processed so far...
                const ICoverageState &coverageState = thresholdsSubset.getCoverageState();
                const IScoreVector &scoreVector = partition.evaluateOutOfSample(thresholdsSubset, coverageState,
                                                                                *bestHead);
                float qualityScore = scoreVector.overallQualityScore;

                // Debugger: print iteration mask
                Debugger::printCoverageMask(coverageState, false, n);
                // Debugger: print quality scores
                Debugger::printQualityScores(bestQualityScore, qualityScore);

                // Check if the quality score is better than the best quality score known so far (reaching the same score
                // with fewer conditions is considered an improvement)...
                if (qualityScore < bestQualityScore || (numPrunedConditions == 0 && qualityScore == bestQualityScore)) {
                    bestQualityScore = qualityScore;
                    bestCoverageStatePtr = coverageState.copy();
                    numPrunedConditions = (numConditions - n);

                    // if the head has more than one element
                    if (bestHead->isPartial()
                        && dynamic_cast<const PartialPrediction *>(bestHead)->getNumElements() >= 2) {
                        // check if a better bestHead exists with pruned conditions
                        const AbstractEvaluatedPrediction *head = scoreVector.processScores(bestHead,
                                                                                            (IScoreProcessor &) *this);
                        if (head != nullptr) {
                            bestHead = head;
                        }
                    }
                }

                // Debugger: print number of pruned conditions
                Debugger::printPrunedConditions(numPrunedConditions);

                conditionIterator++;
            }

            // Remove the pruned conditions...
            while (numPrunedConditions > 0) {
                conditions.removeLast();
                numPrunedConditions--;
            }
        }

        return bestCoverageStatePtr;
    }
}
