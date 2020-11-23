#include "head_refinement_full.h"


/**
 * Allows to find the best multi-label head that predicts for all labels.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels that are considered when searching
 *           for the best head
 */
template<class T>
class FullHeadRefinement : public IHeadRefinement {

    private:

        const T& labelIndices_;

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr_;

    public:

        /**
         * @param labelIndices A reference to an object of template type `T` that provides access to the indices of the
         *                     labels that should be considered when searching for the best head
         */
        FullHeadRefinement(const T& labelIndices)
            : labelIndices_(labelIndices) {

        }

        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated) override {
            const EvaluatedPrediction& prediction = statisticsSubset.calculateExampleWisePrediction(uncovered,
                                                                                                    accumulated);
            float64 overallQualityScore = prediction.overallQualityScore;

            // The quality score must be better than that of `bestHead`...
            if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
                uint32 numPredictions = prediction.getNumElements();
                EvaluatedPrediction::score_const_iterator scoreIterator = prediction.scores_cbegin();

                if (headPtr_.get() == nullptr) {
                    if (labelIndices_.isPartial()) {
                        typename T::const_iterator indexIterator = labelIndices_.cbegin();
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

        std::unique_ptr<AbstractEvaluatedPrediction> pollHead() override {
            return std::move(headPtr_);
        }

        const EvaluatedPrediction& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
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
