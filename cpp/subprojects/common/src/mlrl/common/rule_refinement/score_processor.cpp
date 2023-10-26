#include "mlrl/common/rule_refinement/score_processor.hpp"

#include "mlrl/common/rule_refinement/prediction_complete.hpp"
#include "mlrl/common/rule_refinement/prediction_partial.hpp"
#include "mlrl/common/util/view_functions.hpp"

template<typename T>
static inline void processCompleteScores(std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr, const T& scoreVector) {
    CompletePrediction* existingHead = dynamic_cast<CompletePrediction*>(existingHeadPtr.get());
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<CompletePrediction>(numElements);
        existingHead = (CompletePrediction*) existingHeadPtr.get();
    }

    copyView(scoreVector.values_cbegin(), existingHead->values_begin(), numElements);
    existingHead->quality = scoreVector.quality;
}

template<typename T>
static inline void processPartialScores(std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr, const T& scoreVector) {
    PartialPrediction* existingHead = dynamic_cast<PartialPrediction*>(existingHeadPtr.get());
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<PartialPrediction>(numElements, scoreVector.isSorted());
        existingHead = (PartialPrediction*) existingHeadPtr.get();
    } else {
        // Adjust the size of the existing head, if necessary...
        if (existingHead->getNumElements() != numElements) {
            existingHead->setNumElements(numElements, false);
        }

        existingHead->setSorted(scoreVector.isSorted());
    }

    copyView(scoreVector.values_cbegin(), existingHead->values_begin(), numElements);
    copyView(scoreVector.indices_cbegin(), existingHead->indices_begin(), numElements);
    existingHead->quality = scoreVector.quality;
}

ScoreProcessor::ScoreProcessor(std::unique_ptr<IEvaluatedPrediction>& headPtr) : headPtr_(headPtr) {}

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
