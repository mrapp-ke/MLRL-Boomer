#include "head_refinement.h"
#include <cstdlib>

using namespace seco;


static inline uint32* argsort(const float64* a, uint32 numElements) {
    IndexedValue<float64> tmpArray[numElements];

    for (uint32 i = 0; i < numElements; i++) {
        tmpArray[i].index = i;
        tmpArray[i].value = a[i];
    }

    qsort(&tmpArray, numElements, sizeof(IndexedValue<float64>), &tuples::compareIndexedValue<float64>);
    uint32* sortedArray = new uint32[numElements];

    for (uint32 i = 0; i < numElements; i++) {
        sortedArray[i] = tmpArray[i].index;
    }

    return sortedArray;
}

PartialHeadRefinementImpl::PartialHeadRefinementImpl(std::shared_ptr<ILiftFunction> liftFunctionPtr)
    : liftFunctionPtr_(liftFunctionPtr) {

}

bool PartialHeadRefinementImpl::findHead(const PredictionCandidate* bestHead,
                                         std::unique_ptr<PredictionCandidate>& headPtr, const uint32* labelIndices,
                                         IStatisticsSubset& statisticsSubset, bool uncovered, bool accumulated) {
    bool result = false;
    const LabelWiseEvaluatedPrediction& prediction = statisticsSubset.calculateLabelWisePrediction(uncovered,
                                                                                                   accumulated);
    uint32 numPredictions = prediction.getNumElements();
    LabelWiseEvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();
    LabelWiseEvaluatedPrediction::quality_score_const_iterator qualityScoreIterator =
        prediction.quality_scores_cbegin();
    uint32* sortedIndices = NULL;
    float64 sumOfQualityScores = 0;
    uint32 bestNumPredictions = 0;
    float64 bestQualityScore = 0;

    if (labelIndices == NULL) {
        sortedIndices = argsort(qualityScoreIterator, numPredictions);
        float64 maximumLift = liftFunctionPtr_->getMaxLift();

        for (uint32 c = 0; c < numPredictions; c++) {
            sumOfQualityScores += 1 - qualityScoreIterator[sortedIndices[c]];
            float64 qualityScore = 1 - (sumOfQualityScores / (c + 1)) * liftFunctionPtr_->calculateLift(c + 1);

            if (c == 0 || qualityScore < bestQualityScore) {
                bestNumPredictions = c + 1;
                bestQualityScore = qualityScore;
            }

            if (qualityScore * maximumLift < bestQualityScore) {
                // Prunable by decomposition...
                break;
            }
        }
    } else {
        for (uint32 c = 0; c < numPredictions; c++) {
            sumOfQualityScores += 1 - qualityScoreIterator[c];
        }

        bestQualityScore = 1 - (sumOfQualityScores / numPredictions) * liftFunctionPtr_->calculateLift(numPredictions);
        bestNumPredictions = numPredictions;
    }

    if (bestHead == NULL || bestQualityScore < bestHead->overallQualityScore_) {
        result = true;
        PredictionCandidate* recyclableHead = headPtr.get();

        if (recyclableHead == NULL) {
            // Create a new `PredictionCandidate`...
            uint32* candidateLabelIndices = (uint32*) malloc(bestNumPredictions * sizeof(uint32));
            float64* candidatePredictedScores = (float64*) malloc(bestNumPredictions * sizeof(float64));

            if (labelIndices == NULL) {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    uint32 i = sortedIndices[c];
                    candidateLabelIndices[c] = labelIndices == NULL ? i : labelIndices[i];
                    candidatePredictedScores[c] = valueIterator[i];
                }
            } else {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    candidateLabelIndices[c] = labelIndices[c];
                    candidatePredictedScores[c] = valueIterator[c];
                }
            }

            headPtr = std::make_unique<PredictionCandidate>(bestNumPredictions, candidateLabelIndices,
                                                            candidatePredictedScores, bestQualityScore);
        } else {
            // Modify the `recyclableHead`...
            if (recyclableHead->numPredictions_ != bestNumPredictions) {
                recyclableHead->numPredictions_ = bestNumPredictions;
                recyclableHead->labelIndices_ = (uint32*) realloc(recyclableHead->labelIndices_,
                                                                  bestNumPredictions * sizeof(uint32));
                recyclableHead->predictedScores_ = (float64*) realloc(recyclableHead->predictedScores_,
                                                                      bestNumPredictions * sizeof(float64));
            }

            if (labelIndices == NULL) {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    uint32 i = sortedIndices[c];
                    recyclableHead->labelIndices_[c] = labelIndices == NULL ? i : labelIndices[i];
                    recyclableHead->predictedScores_[c] = valueIterator[i];
                }
            } else {
                for (uint32 c = 0; c < bestNumPredictions; c++) {
                    recyclableHead->labelIndices_[c] = labelIndices[c];
                    recyclableHead->predictedScores_[c] = valueIterator[c];
                }
            }

            recyclableHead->overallQualityScore_ = bestQualityScore;
        }
    }

    delete[] sortedIndices;
    return result;
}

std::unique_ptr<PredictionCandidate> PartialHeadRefinementImpl::pollHead() {
    // TODO
}

const EvaluatedPrediction& PartialHeadRefinementImpl::calculatePrediction(IStatisticsSubset& statisticsSubset,
                                                                          bool uncovered, bool accumulated) const {
    return statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
}

PartialHeadRefinementFactoryImpl::PartialHeadRefinementFactoryImpl(std::shared_ptr<ILiftFunction> liftFunctionPtr)
    : liftFunctionPtr_(liftFunctionPtr) {

}

std::unique_ptr<IHeadRefinement> PartialHeadRefinementFactoryImpl::create() const {
    return std::make_unique<PartialHeadRefinementImpl>(liftFunctionPtr_);
}
