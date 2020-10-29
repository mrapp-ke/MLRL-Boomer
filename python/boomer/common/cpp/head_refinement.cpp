#include "head_refinement.h"
#include <cstdlib>


/**
 * Allows to find the best single-label head that predicts for a single label.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels that are considered when searching
 *           for the best head
 */
template<class T>
class SingleLabelHeadRefinement : virtual public IHeadRefinement {

    private:

        const T& labelIndices_;

        std::unique_ptr<PartialPrediction> headPtr_;

    public:

        /**
         * @param labelIndices A reference to an object of template type `T` that provides access to the indices of the
         *                     labels that should be considered when searching for the best head
         */
        SingleLabelHeadRefinement(const T& labelIndices)
            : labelIndices_(labelIndices) {

        }

        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated) override {
            const LabelWiseEvaluatedPrediction& prediction = statisticsSubset.calculateLabelWisePrediction(uncovered,
                                                                                                           accumulated);
            uint32 numPredictions = prediction.getNumElements();
            LabelWiseEvaluatedPrediction::quality_score_const_iterator qualityScoreIterator =
                prediction.quality_scores_cbegin();
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
                LabelWiseEvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();
                typename T::index_const_iterator indexIterator = labelIndices_.indices_cbegin();

                if (headPtr_.get() == nullptr) {
                    headPtr_ = std::make_unique<PartialPrediction>(1);
                }

                PartialPrediction::iterator headValueIterator = headPtr_->begin();
                PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();
                headValueIterator[0] = valueIterator[bestC];
                headIndexIterator[0] = indexIterator[bestC];
                headPtr_->overallQualityScore = bestQualityScore;
                return headPtr_.get();
            }

            return nullptr;
        }

        std::unique_ptr<AbstractEvaluatedPrediction> pollHead() override {
            return std::move(headPtr_);
        }

        const EvaluatedPrediction& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                       bool accumulated) const override {
            return statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
        }

};

/**
 * Allows to find the best multi-label head that predicts for all labels.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels that are considered when searching
 *           for the best head
 */
template<class T>
class FullHeadRefinement : virtual public IHeadRefinement {

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
            const EvaluatedPrediction& prediction = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
            float64 overallQualityScore = prediction.overallQualityScore;

            // The quality score must be better than that of `bestHead`...
            if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
                uint32 numPredictions = prediction.getNumElements();
                EvaluatedPrediction::const_iterator valueIterator = prediction.cbegin();

                if (headPtr_.get() == nullptr) {
                    if (labelIndices_.isPartial()) {
                        typename T::index_const_iterator indexIterator = labelIndices_.indices_cbegin();
                        std::unique_ptr<PartialPrediction> headPtr = std::make_unique<PartialPrediction>(numPredictions);
                        PartialPrediction::index_iterator headIndexIterator = headPtr->indices_begin();

                        for (uint32 c = 0; c < numPredictions; c++) {
                            headIndexIterator[c] = indexIterator[c];
                        }

                        headPtr_ = std::move(headPtr);
                    } else {
                        headPtr_ = std::make_unique<FullPrediction>(numPredictions);
                    }
                }

                AbstractEvaluatedPrediction::iterator headValueIterator = headPtr_->begin();

                for (uint32 c = 0; c < numPredictions; c++) {
                    headValueIterator[c] = valueIterator[c];
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

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactoryImpl::create(
        const FullIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinement<FullIndexVector>>(labelIndices);
}

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactoryImpl::create(
        const PartialIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinement<PartialIndexVector>>(labelIndices);
}

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactoryImpl::create(const FullIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinement<FullIndexVector>>(labelIndices);
}

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactoryImpl::create(const PartialIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinement<PartialIndexVector>>(labelIndices);
}
