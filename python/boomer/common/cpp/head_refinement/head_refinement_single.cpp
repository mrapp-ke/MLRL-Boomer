#include "head_refinement_single.h"
#include "prediction_partial.h"
#include "../rule_evaluation/score_processor_label_wise.h"


/**
 * Allows to find the best single-label head that predicts for a single label.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels that are considered when searching
 *           for the best head
 */
template<class T>
class SingleLabelHeadRefinement : public IHeadRefinement, public ILabelWiseScoreProcessor {

    private:

        const T& labelIndices_;

        std::unique_ptr<PartialPrediction> headPtr_;

        template<class T2>
        const AbstractEvaluatedPrediction* processScoresInternally(const AbstractEvaluatedPrediction* bestHead,
                                                                   const T2& scoreVector) {
            uint32 numPredictions = scoreVector.getNumElements();
            typename T2::quality_score_const_iterator qualityScoreIterator = scoreVector.quality_scores_cbegin();
            uint32 bestC = 0;
            float64 bestQualityScore = qualityScoreIterator[bestC];

            for (uint32 c = 1; c < numPredictions; c++) {
                float64 qualityScore = qualityScoreIterator[c];

                if (qualityScore < bestQualityScore) {
                    bestQualityScore = qualityScore;
                    bestC = c;
                }
            }

            // The quality score must be better than that of `bestHead`...
            if (bestHead == nullptr || bestQualityScore < bestHead->overallQualityScore) {
                typename T2::score_const_iterator scoreIterator = scoreVector.scores_cbegin();
                typename T::const_iterator indexIterator = labelIndices_.cbegin();

                if (headPtr_.get() == nullptr) {
                    headPtr_ = std::make_unique<PartialPrediction>(1);
                }

                PartialPrediction::score_iterator headScoreIterator = headPtr_->scores_begin();
                PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();
                headScoreIterator[0] = scoreIterator[bestC];
                headIndexIterator[0] = indexIterator[bestC];
                headPtr_->overallQualityScore = bestQualityScore;
                return headPtr_.get();
            }

            return nullptr;
        }

    public:

        /**
         * @param labelIndices A reference to an object of template type `T` that provides access to the indices of the
         *                     labels that should be considered when searching for the best head
         */
        SingleLabelHeadRefinement(const T& labelIndices)
            : labelIndices_(labelIndices) {

        }

        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         const DenseLabelWiseScoreVector& scoreVector) {
            return processScoresInternally<DenseLabelWiseScoreVector>(bestHead, scoreVector);
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

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactory::create(
        const FullIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinement<FullIndexVector>>(labelIndices);
}

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactory::create(
        const PartialIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinement<PartialIndexVector>>(labelIndices);
}
