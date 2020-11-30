#include "head_refinement_full.h"
#include "prediction_full.h"
#include "prediction_partial.h"
#include "../rule_evaluation/score_processor.h"


/**
 * Allows to find the best multi-label head that predicts for all labels.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels that are considered when searching
 *           for the best head
 */
template<class T>
class FullHeadRefinement : public IHeadRefinement, public IScoreProcessor {

    private:

        const T& labelIndices_;

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr_;

        template<class T2>
        const AbstractEvaluatedPrediction* processScoresInternally(const AbstractEvaluatedPrediction* bestHead,
                                                                   const T2& scoreVector) {
            float64 overallQualityScore = scoreVector.overallQualityScore;

            // The quality score must be better than that of `bestHead`...
            if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
                uint32 numPredictions = scoreVector.getNumElements();
                typename T2::score_const_iterator scoreIterator = scoreVector.scores_cbegin();

                if (headPtr_.get() == nullptr) {
                    if (scoreVector.isPartial()) {
                        typename T2::index_const_iterator indexIterator = scoreVector.indices_cbegin();
                        std::unique_ptr<PartialPrediction> headPtr =
                            std::make_unique<PartialPrediction>(numPredictions);
                        PartialPrediction::index_iterator headIndexIterator = headPtr->indices_begin();

                        for (uint32 c = 0; c < numPredictions; c++) {
                            headIndexIterator[c] = indexIterator[c];
                        }

                        headPtr_ = std::move(headPtr);
                    } else {
                        headPtr_ = std::make_unique<FullPrediction>(numPredictions);
                    }
                }

                AbstractEvaluatedPrediction::score_iterator headScoreIterator = headPtr_->scores_begin();

                for (uint32 c = 0; c < numPredictions; c++) {
                    headScoreIterator[c] = scoreIterator[c];
                }

                headPtr_->overallQualityScore = overallQualityScore;
                return headPtr_.get();
            }

            return nullptr;
        }

    public:

        /**
         * @param labelIndices A reference to an object of template type `T` that provides access to the indices of the
         *                     labels that should be considered when searching for the best head
         */
        FullHeadRefinement(const T& labelIndices)
            : labelIndices_(labelIndices) {

        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseScoreVector<FullIndexVector>& scoreVector) override {
            return processScoresInternally<DenseScoreVector<FullIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseScoreVector<PartialIndexVector>& scoreVector) override {
            return processScoresInternally<DenseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated) override {
            const IScoreVector& scoreVector = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
            return scoreVector.processScores(bestHead, *this);
        }

        std::unique_ptr<AbstractEvaluatedPrediction> pollHead() override {
            return std::move(headPtr_);
        }

        const IScoreVector& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                bool accumulated) const override {
            return statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
        }

};

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactory::create(const FullIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinement<FullIndexVector>>(labelIndices);
}

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactory::create(const PartialIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinement<PartialIndexVector>>(labelIndices);
}
