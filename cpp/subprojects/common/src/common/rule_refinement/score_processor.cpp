#include "common/rule_refinement/score_processor.hpp"
#include "common/rule_refinement/prediction_complete.hpp"
#include "common/rule_refinement/prediction_partial.hpp"
#include "common/data/arrays.hpp"


template<typename T>
static inline void processCompleteScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                         const T& scoreVector) {
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHeadPtr) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<CompletePrediction>(numElements);
    }

    copyArray(scoreVector.scores_cbegin(), existingHeadPtr->scores_begin(), numElements);
    existingHeadPtr->overallQualityScore = scoreVector.overallQualityScore;
}

template<typename T>
static inline void processPartialScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                        const T& scoreVector) {
    PartialPrediction* existingHead = (PartialPrediction*) existingHeadPtr.get();
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<PartialPrediction>(numElements);
        existingHead = (PartialPrediction*) existingHeadPtr.get();
    } else if (existingHead->getNumElements() != numElements) {
        // Adjust the size of the existing head, if necessary...
        existingHead->setNumElements(numElements, false);
    }

    copyArray(scoreVector.scores_cbegin(), existingHead->scores_begin(), numElements);
    copyArray(scoreVector.indices_cbegin(), existingHead->indices_begin(), numElements);
    existingHead->overallQualityScore = scoreVector.overallQualityScore;
}

void ScoreProcessor::processScores(const DenseScoreVector<CompleteIndexVector>& scoreVector) {
    processCompleteScores(headPtr_, scoreVector);
}

void ScoreProcessor::processScores(const DenseScoreVector<PartialIndexVector>& scoreVector) {
    processPartialScores(headPtr_, scoreVector);
}

void ScoreProcessor::processScores(const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) {
    processCompleteScores(headPtr_, scoreVector);
}

void ScoreProcessor::processScores(const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) {
    processPartialScores(headPtr_, scoreVector);
}

void ScoreProcessor::processScores(const IScoreVector& scoreVector) {
    scoreVector.processScores(*this);
}

std::unique_ptr<AbstractEvaluatedPrediction> ScoreProcessor::pollHead() {
    return std::move(headPtr_);
}
