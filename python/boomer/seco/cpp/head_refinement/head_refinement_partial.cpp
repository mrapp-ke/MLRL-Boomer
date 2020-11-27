#include "head_refinement_partial.h"
#include "../../../common/cpp/data/vector_sparse_array.h"
#include "../../../common/cpp/head_refinement/prediction_partial.h"
#include "../../../common/cpp/rule_evaluation/score_processor_label_wise.h"

using namespace seco;


static inline std::unique_ptr<SparseArrayVector<float64>> argsort(const float64* a, uint32 numElements) {
    std::unique_ptr<SparseArrayVector<float64>> sortedVectorPtr = std::make_unique<SparseArrayVector<float64>>(
        numElements);
    SparseArrayVector<float64>::iterator iterator = sortedVectorPtr->begin();

    for (uint32 i = 0; i < numElements; i++) {
        iterator[i].index = i;
        iterator[i].value = a[i];
    }

    sortedVectorPtr->sortByValues();
    return sortedVectorPtr;
}

/**
 * Allows to find the best head that predicts for one or several labels depending on a lift function.
 */
class PartialHeadRefinement : public IHeadRefinement, public ILabelWiseScoreProcessor {

    private:

        std::shared_ptr<ILiftFunction> liftFunctionPtr_;

        std::unique_ptr<PartialPrediction> headPtr_;

        template<class T>
        const AbstractEvaluatedPrediction* processScoresInternally(const AbstractEvaluatedPrediction* bestHead,
                                                                   const T scoreVector) {
            uint32 numPredictions = scoreVector.getNumElements();
            typename T::score_const_iterator scoreIterator = scoreVector.scores_cbegin();
            typename T::quality_score_const_iterator qualityScoreIterator = scoreVector.quality_scores_cbegin();
            std::unique_ptr<SparseArrayVector<float64>> sortedVectorPtr;
            float64 sumOfQualityScores = 0;
            uint32 bestNumPredictions = 0;
            float64 bestQualityScore = 0;

            if (scoreVector.isPartial()) {
                for (uint32 c = 0; c < numPredictions; c++) {
                    sumOfQualityScores += 1 - qualityScoreIterator[c];
                }

                bestQualityScore =
                    1 - (sumOfQualityScores / numPredictions) * liftFunctionPtr_->calculateLift(numPredictions);
                bestNumPredictions = numPredictions;
            } else {
                sortedVectorPtr = argsort(qualityScoreIterator, numPredictions);
                SparseArrayVector<float64>::const_iterator sortedIterator = sortedVectorPtr->cbegin();
                float64 maximumLift = liftFunctionPtr_->getMaxLift();

                for (uint32 c = 0; c < numPredictions; c++) {
                    sumOfQualityScores += 1 - qualityScoreIterator[sortedIterator[c].index];
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
            }

            if (bestHead == nullptr || bestQualityScore < bestHead->overallQualityScore) {
                if (headPtr_.get() == nullptr) {
                    headPtr_ = std::make_unique<PartialPrediction>(bestNumPredictions);
                } else if (headPtr_->getNumElements() != bestNumPredictions) {
                    headPtr_->setNumElements(bestNumPredictions, false);
                }

                typename T::index_const_iterator indexIterator = scoreVector.indices_cbegin();
                PartialPrediction::score_iterator headScoreIterator = headPtr_->scores_begin();
                PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();

                if (scoreVector.isPartial()) {
                    for (uint32 c = 0; c < bestNumPredictions; c++) {
                        headIndexIterator[c] = indexIterator[c];
                        headScoreIterator[c] = scoreIterator[c];
                    }
                } else {
                    SparseArrayVector<float64>::const_iterator sortedIterator = sortedVectorPtr->cbegin();

                    for (uint32 c = 0; c < bestNumPredictions; c++) {
                        uint32 i = sortedIterator[c].index;
                        headIndexIterator[c] = indexIterator[i];
                        headScoreIterator[c] = scoreIterator[i];
                    }
                }

                headPtr_->overallQualityScore = bestQualityScore;
                return headPtr_.get();
            }

            return nullptr;
        }

    public:

        /**
         * @param liftFunctionPtr A shared pointer to an object of type `ILiftFunction` that should affect the quality
         *                        scores of rules, depending on how many labels they predict
         */
        PartialHeadRefinement(std::shared_ptr<ILiftFunction> liftFunctionPtr)
            : liftFunctionPtr_(liftFunctionPtr) {

        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseLabelWiseScoreVector<FullIndexVector>& scoreVector) {
            return processScoresInternally<DenseLabelWiseScoreVector<FullIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseLabelWiseScoreVector<PartialIndexVector>& scoreVector) {
            return processScoresInternally<DenseLabelWiseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated) override {
            const ILabelWiseScoreVector& scoreVector = statisticsSubset.calculateLabelWiseScores(uncovered,
                                                                                                 accumulated);
            return scoreVector.processScores(bestHead, *this);
        }

        std::unique_ptr<AbstractEvaluatedPrediction> pollHead() override {
            return std::move(headPtr_);
        }

        const IScoreVector& calculateScores(IStatisticsSubset& statisticsSubset, bool uncovered,
                                            bool accumulated) const override {
            return statisticsSubset.calculateLabelWiseScores(uncovered, accumulated);
        }

};

PartialHeadRefinementFactory::PartialHeadRefinementFactory(std::shared_ptr<ILiftFunction> liftFunctionPtr)
    : liftFunctionPtr_(liftFunctionPtr) {

}

std::unique_ptr<IHeadRefinement> PartialHeadRefinementFactory::create() const {
    return std::make_unique<PartialHeadRefinement>(liftFunctionPtr_);
}
