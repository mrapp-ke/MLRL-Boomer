#include "mlrl/common/rule_refinement/score_processor.hpp"

#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "mlrl/common/rule_refinement/prediction_complete.hpp"
#include "mlrl/common/rule_refinement/prediction_partial.hpp"

template<typename ScoreVector>
static inline void processCompleteScores(std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr,
                                         const ScoreVector& scoreVector,
                                         IStatisticsUpdateFactory& statisticsUpdateFactory) {
    CompletePrediction* existingHead = dynamic_cast<CompletePrediction*>(existingHeadPtr.get());
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<CompletePrediction>(numElements, statisticsUpdateFactory);
        existingHead = static_cast<CompletePrediction*>(existingHeadPtr.get());
    }

    util::copyView(scoreVector.values_cbegin(), existingHead->values_begin(), numElements);
    existingHead->quality = scoreVector.quality;
}

template<typename ScoreVector>
static inline void processPartialScores(std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr,
                                        const ScoreVector& scoreVector,
                                        IStatisticsUpdateFactory& statisticsUpdateFactory) {
    PartialPrediction* existingHead = dynamic_cast<PartialPrediction*>(existingHeadPtr.get());
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr =
          std::make_unique<PartialPrediction>(numElements, scoreVector.isSorted(), statisticsUpdateFactory);
        existingHead = static_cast<PartialPrediction*>(existingHeadPtr.get());
    } else {
        // Adjust the size of the existing head, if necessary...
        if (existingHead->getNumElements() != numElements) {
            existingHead->setNumElements(statisticsUpdateFactory, numElements, false);
        }

        existingHead->setSorted(scoreVector.isSorted());
    }

    util::copyView(scoreVector.values_cbegin(), existingHead->values_begin(), numElements);
    util::copyView(scoreVector.indices_cbegin(), existingHead->indices_begin(), numElements);
    existingHead->quality = scoreVector.quality;
}

ScoreProcessor::ScoreProcessor(std::unique_ptr<IEvaluatedPrediction>& headPtr) : headPtr_(headPtr) {}

void ScoreProcessor::processScores(const StatisticsUpdateCandidate& scores) {
    auto completeDense32BitVisitor = [this](const DenseScoreVector<float32, CompleteIndexVector>& scoreVector,
                                            IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDense32BitVisitor = [this](const DenseScoreVector<float32, PartialIndexVector>& scoreVector,
                                           IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto completeDense64BitVisitor = [this](const DenseScoreVector<float64, CompleteIndexVector>& scoreVector,
                                            IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDense64BitVisitor = [this](const DenseScoreVector<float64, PartialIndexVector>& scoreVector,
                                           IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto completeDenseBinned32BitVisitor = [this](
                                             const DenseBinnedScoreVector<float32, CompleteIndexVector>& scoreVector,
                                             IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDenseBinned32BitVisitor = [this](const DenseBinnedScoreVector<float32, PartialIndexVector>& scoreVector,
                                                 IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto completeDenseBinned64BitVisitor = [this](
                                             const DenseBinnedScoreVector<float64, CompleteIndexVector>& scoreVector,
                                             IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDenseBinned64BitVisitor = [this](const DenseBinnedScoreVector<float64, PartialIndexVector>& scoreVector,
                                                 IStatisticsUpdateFactory& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    scores.visit(completeDense32BitVisitor, partialDense32BitVisitor, completeDense64BitVisitor,
                 partialDense64BitVisitor, completeDenseBinned32BitVisitor, partialDenseBinned32BitVisitor,
                 completeDenseBinned64BitVisitor, partialDenseBinned64BitVisitor);
}
