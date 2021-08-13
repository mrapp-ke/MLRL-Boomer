#include "common/rule_evaluation/score_processor.hpp"
#include "common/head_refinement/prediction_complete.hpp"
#include "common/head_refinement/prediction_partial.hpp"
#include <algorithm>


template<typename T>
const AbstractEvaluatedPrediction* processScoresInternally(
        std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr, const AbstractEvaluatedPrediction* bestHead,
        const T& scoreVector) {
    float64 overallQualityScore = scoreVector.overallQualityScore;

    // The quality score must be better than that of `bestHead`...
    if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
        uint32 numPredictions = scoreVector.getNumElements();

        if (existingHeadPtr.get() == nullptr) {
            // Create a new head, if necessary...
            if (scoreVector.isPartial()) {
                std::unique_ptr<PartialPrediction> headPtr = std::make_unique<PartialPrediction>(numPredictions);
                existingHeadPtr = std::move(headPtr);
            } else {
                existingHeadPtr = std::make_unique<CompletePrediction>(numPredictions);
            }
        } else if (existingHeadPtr->getNumElements() != numPredictions) {
            // Adjust the size of the existing head, if necessary...
            existingHeadPtr->setNumElements(numPredictions, false);
        }

        // TODO This is a hack
        PartialPrediction* partialHead = dynamic_cast<PartialPrediction*>(existingHeadPtr.get());
        if (partialHead != nullptr) {
            std::copy(scoreVector.indices_cbegin(), scoreVector.indices_cend(), partialHead->indices_begin());
        }

        std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), existingHeadPtr->scores_begin());
        existingHeadPtr->overallQualityScore = overallQualityScore;
        return existingHeadPtr.get();
    }

    return nullptr;
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseScoreVector<CompleteIndexVector>& scoreVector) {
    return processScoresInternally<DenseScoreVector<CompleteIndexVector>>(headPtr_, bestHead, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseScoreVector<PartialIndexVector>& scoreVector) {
    return processScoresInternally<DenseScoreVector<PartialIndexVector>>(headPtr_, bestHead, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) {
    return processScoresInternally<DenseBinnedScoreVector<CompleteIndexVector>>(headPtr_, bestHead, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const AbstractEvaluatedPrediction* bestHead, const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) {
    return processScoresInternally<DenseBinnedScoreVector<PartialIndexVector>>(headPtr_, bestHead, scoreVector);
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
