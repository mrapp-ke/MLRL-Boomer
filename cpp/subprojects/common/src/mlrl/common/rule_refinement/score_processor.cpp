#include "mlrl/common/rule_refinement/score_processor.hpp"

#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "mlrl/common/rule_refinement/prediction_complete.hpp"
#include "mlrl/common/rule_refinement/prediction_partial.hpp"

template<typename ScoreVector>
static inline void processCompleteScores(std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr,
                                         const ScoreVector& scoreVector) {
    CompletePrediction* existingHead = dynamic_cast<CompletePrediction*>(existingHeadPtr.get());
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<CompletePrediction>(numElements);
        existingHead = static_cast<CompletePrediction*>(existingHeadPtr.get());
    }

    util::copyView(scoreVector.values_cbegin(), existingHead->values_begin(), numElements);
    existingHead->quality = scoreVector.quality;
}

template<typename ScoreVector>
static inline void processPartialScores(std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr,
                                        const ScoreVector& scoreVector) {
    PartialPrediction* existingHead = dynamic_cast<PartialPrediction*>(existingHeadPtr.get());
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<PartialPrediction>(numElements, scoreVector.isSorted());
        existingHead = static_cast<PartialPrediction*>(existingHeadPtr.get());
    } else {
        // Adjust the size of the existing head, if necessary...
        if (existingHead->getNumElements() != numElements) {
            existingHead->setNumElements(numElements, false);
        }

        existingHead->setSorted(scoreVector.isSorted());
    }

    util::copyView(scoreVector.values_cbegin(), existingHead->values_begin(), numElements);
    util::copyView(scoreVector.indices_cbegin(), existingHead->indices_begin(), numElements);
    existingHead->quality = scoreVector.quality;
}

ScoreProcessor::ScoreProcessor(std::unique_ptr<IEvaluatedPrediction>& headPtr) : headPtr_(headPtr) {}

void ScoreProcessor::processScores(const StatisticsUpdateCandidate& scores) {
    auto completeDenseVisitor = [this](const DenseScoreVector<CompleteIndexVector>& scoreVector,
                                       IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector);
    };
    auto partialDenseVisitor = [this](const DenseScoreVector<PartialIndexVector>& scoreVector,
                                      IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector);
    };
    auto completeDenseBinnedVisitor = [this](const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector,
                                             IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector);
    };
    auto partialDenseBinnedVisitor = [this](const DenseBinnedScoreVector<PartialIndexVector>& scoreVector,
                                            IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector);
    };
    scores.visit(completeDenseVisitor, partialDenseVisitor, completeDenseBinnedVisitor, partialDenseBinnedVisitor);
}
