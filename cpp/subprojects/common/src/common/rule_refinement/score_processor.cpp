#include "common/rule_refinement/score_processor.hpp"
#include "common/rule_refinement/prediction_complete.hpp"
#include "common/rule_refinement/prediction_partial.hpp"
#include <algorithm>


template<typename T>
const AbstractEvaluatedPrediction* processCompleteScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                                         const AbstractEvaluatedPrediction* bestHead,
                                                         const T& scoreVector) {
    float64 overallQualityScore = scoreVector.overallQualityScore;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
        if (existingHeadPtr.get() == nullptr) {
            // Create a new head, if necessary...
            uint32 numElements = scoreVector.getNumElements();
            existingHeadPtr = std::make_unique<CompletePrediction>(numElements);
        }

        std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), existingHeadPtr->scores_begin());
        existingHeadPtr->overallQualityScore = overallQualityScore;
        return existingHeadPtr.get();
    }

    return nullptr;
}

template<typename T>
const AbstractEvaluatedPrediction* processPartialScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                                        const AbstractEvaluatedPrediction* bestHead,
                                                        const T& scoreVector) {
    float64 overallQualityScore = scoreVector.overallQualityScore;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
        PartialPrediction* existingHead = (PartialPrediction*) existingHeadPtr.get();

        if (existingHead == nullptr) {
            // Create a new head, if necessary...
            uint32 numElements = scoreVector.getNumElements();
            existingHeadPtr = std::make_unique<PartialPrediction>(numElements);
            existingHead = (PartialPrediction*) existingHeadPtr.get();
        } else {
            // Adjust the size of the existing head, if necessary...
            uint32 numElements = scoreVector.getNumElements();

            if (existingHead->getNumElements() != numElements) {
                existingHead->setNumElements(numElements, false);
            }
        }

        std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), existingHead->scores_begin());
        std::copy(scoreVector.indices_cbegin(), scoreVector.indices_cend(), existingHead->indices_begin());
        existingHead->overallQualityScore = overallQualityScore;
        return existingHead;
    }

    return nullptr;
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseScoreVector<CompleteIndexVector>& scoreVector) {
    return processCompleteScores(headPtr_, bestHead, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseScoreVector<PartialIndexVector>& scoreVector) {
    return processPartialScores(headPtr_, bestHead, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) {
    return processCompleteScores(headPtr_, bestHead, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) {
    return processPartialScores(headPtr_, bestHead, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::findHead(const AbstractEvaluatedPrediction* bestHead,
                                                            IStatisticsSubset& statisticsSubset, bool uncovered,
                                                            bool accumulated) {
    const IScoreVector& scoreVector = statisticsSubset.calculatePrediction(uncovered, accumulated);
    return scoreVector.processScores(bestHead, *this);
}

std::unique_ptr<AbstractEvaluatedPrediction> ScoreProcessor::pollHead() {
    return std::move(headPtr_);
}
