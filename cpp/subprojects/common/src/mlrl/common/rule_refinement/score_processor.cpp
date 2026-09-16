#include "mlrl/common/rule_refinement/score_processor.hpp"

#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"
#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"
#include "mlrl/common/rule_refinement/prediction_complete.hpp"
#include "mlrl/common/rule_refinement/prediction_partial.hpp"

template<typename ScoreVector>
static inline void processCompleteScores(
  std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr, const ScoreVector& scoreVector,
  IStatisticsUpdateFactory<typename ScoreVector::score_type>& statisticsUpdateFactory) {
    using score_type = ScoreVector::score_type;
    CompletePrediction<score_type>* existingHead = dynamic_cast<CompletePrediction<score_type>*>(existingHeadPtr.get());

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr =
          std::make_unique<CompletePrediction<score_type>>(scoreVector.getNumElements(), statisticsUpdateFactory);
        existingHead = static_cast<CompletePrediction<score_type>*>(existingHeadPtr.get());
    }

    std::copy(scoreVector.cbegin(), scoreVector.cend(), existingHead->values_begin());
    existingHead->quality = scoreVector.quality;
}

template<typename ScoreVector>
static inline void processPartialScores(
  std::unique_ptr<IEvaluatedPrediction>& existingHeadPtr, const ScoreVector& scoreVector,
  IStatisticsUpdateFactory<typename ScoreVector::score_type>& statisticsUpdateFactory) {
    using score_type = ScoreVector::score_type;
    PartialPrediction<score_type>* existingHead = dynamic_cast<PartialPrediction<score_type>*>(existingHeadPtr.get());
    uint32 numElements = scoreVector.getNumElements();

    if (!existingHead) {
        // Create a new head, if necessary...
        existingHeadPtr =
          std::make_unique<PartialPrediction<score_type>>(numElements, scoreVector.isSorted(), statisticsUpdateFactory);
        existingHead = static_cast<PartialPrediction<score_type>*>(existingHeadPtr.get());
    } else {
        // Adjust the size of the existing head, if necessary...
        if (existingHead->getNumElements() != numElements) {
            existingHead->setNumElements(statisticsUpdateFactory, numElements, false);
        }

        existingHead->setSorted(scoreVector.isSorted());
    }

    std::copy(scoreVector.cbegin(), scoreVector.cend(), existingHead->values_begin());
    std::copy(scoreVector.indices_cbegin(), scoreVector.indices_cend(), existingHead->indices_begin());
    existingHead->quality = scoreVector.quality;
}

ScoreProcessor::ScoreProcessor(std::unique_ptr<IEvaluatedPrediction>& headPtr) : headPtr_(headPtr) {}

void ScoreProcessor::processScores(const IStatisticsUpdateCandidate& scores) {
    auto completeBitVisitor = [this](const BitScoreVectorView<CompleteIndexVector>& scoreVector,
                                     IStatisticsUpdateFactory<uint8>& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialBitVisitor = [this](const BitScoreVectorView<PartialIndexVector>& scoreVector,
                                    IStatisticsUpdateFactory<uint8>& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto completeDense32BitVisitor = [this](const DenseScoreVectorView<float32, CompleteIndexVector>& scoreVector,
                                            IStatisticsUpdateFactory<float32>& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDense32BitVisitor = [this](const DenseScoreVectorView<float32, PartialIndexVector>& scoreVector,
                                           IStatisticsUpdateFactory<float32>& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto completeDense64BitVisitor = [this](const DenseScoreVectorView<float64, CompleteIndexVector>& scoreVector,
                                            IStatisticsUpdateFactory<float64>& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDense64BitVisitor = [this](const DenseScoreVectorView<float64, PartialIndexVector>& scoreVector,
                                           IStatisticsUpdateFactory<float64>& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto completeDenseBinned32BitVisitor =
      [this](const DenseBinnedScoreVectorView<float32, CompleteIndexVector>& scoreVector,
             IStatisticsUpdateFactory<float32>& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDenseBinned32BitVisitor = [this](
                                            const DenseBinnedScoreVectorView<float32, PartialIndexVector>& scoreVector,
                                            IStatisticsUpdateFactory<float32>& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto completeDenseBinned64BitVisitor =
      [this](const DenseBinnedScoreVectorView<float64, CompleteIndexVector>& scoreVector,
             IStatisticsUpdateFactory<float64>& statisticsUpdateFactory) {
        processCompleteScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    auto partialDenseBinned64BitVisitor = [this](
                                            const DenseBinnedScoreVectorView<float64, PartialIndexVector>& scoreVector,
                                            IStatisticsUpdateFactory<float64>& statisticsUpdateFactory) {
        processPartialScores(headPtr_, scoreVector, statisticsUpdateFactory);
    };
    scores.visit(completeBitVisitor, partialBitVisitor, completeDense32BitVisitor, partialDense32BitVisitor,
                 completeDense64BitVisitor, partialDense64BitVisitor, completeDenseBinned32BitVisitor,
                 partialDenseBinned32BitVisitor, completeDenseBinned64BitVisitor, partialDenseBinned64BitVisitor);
}
