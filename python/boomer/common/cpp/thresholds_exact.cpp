#include "thresholds_exact.h"
#include "thresholds_common.cpp"
#include "rule_refinement_exact.cpp"


/**
 * Provides access to a subset of the thresholds that are stored by an instance of the class `ExactThresholds`.
 */
class ExactThresholds::ThresholdsSubset : public IThresholdsSubset {

    private:

    /**
     * A callback that allows to retrieve feature vectors. If available, the feature vectors are retrieved from the
     * cache. Otherwise, they are fetched from the feature matrix.
     */
    class Callback : public IRuleRefinementCallback<FeatureVector> {

        private:

            ThresholdsSubset& thresholdsSubset_;

            uint32 featureIndex_;

        public:

            /**
             * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the feature
             *                          vectors
             * @param featureIndex      The index of the feature for which the feature vector should be retrieved
             */
            Callback(ThresholdsSubset& thresholdsSubset, uint32 featureIndex)
                : thresholdsSubset_(thresholdsSubset), featureIndex_(featureIndex) {

            }

            std::unique_ptr<Result> get() override {
                auto cacheFilteredIterator = thresholdsSubset_.cacheFiltered_.find(featureIndex_);
                FilteredCacheEntry<FeatureVector>& cacheEntry = cacheFilteredIterator->second;
                FeatureVector* featureVector = cacheEntry.vectorPtr.get();

                if (featureVector == nullptr) {
                    auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                    featureVector = cacheIterator->second.get();

                    if (featureVector == nullptr) {
                        thresholdsSubset_.thresholds_.featureMatrixPtr_->fetchFeatureVector(featureIndex_,
                                                                                            cacheIterator->second);
                        cacheIterator->second->sortByValues();
                        featureVector = cacheIterator->second.get();
                    }
                }

                // Filter feature vector, if only a subset of its elements are covered by the current rule...
                uint32 numConditions = thresholdsSubset_.numModifications_;

                if (numConditions > cacheEntry.numConditions) {
                    filterAnyVector<FeatureVector>(*featureVector, cacheEntry, numConditions,
                                                   thresholdsSubset_.coverageMask_);
                    featureVector = cacheEntry.vectorPtr.get();
                }

                return std::make_unique<Result>(*thresholdsSubset_.thresholds_.statisticsPtr_, *featureVector);
            }

    };

    ExactThresholds& thresholds_;

    const IWeightVector& weights_;

    uint32 sumOfWeights_;

    CoverageMask coverageMask_;

    uint32 numModifications_;

    std::unordered_map<uint32, FilteredCacheEntry<FeatureVector>> cacheFiltered_;

    template<class T>
    std::unique_ptr<IRuleRefinement> createExactRuleRefinement(const T& labelIndices, uint32 featureIndex) {
        // Retrieve the `FilteredCacheEntry` from the cache, or insert a new one if it does not already exist...
        auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry<FeatureVector>()).first;
        FeatureVector* featureVector = cacheFilteredIterator->second.vectorPtr.get();

        // If the `FilteredCacheEntry` in the cache does not refer to a `FeatureVector`, add an empty `unique_ptr` to
        // the cache...
        if (featureVector == nullptr) {
            thresholds_.cache_.emplace(featureIndex, std::unique_ptr<FeatureVector>());
        }

        bool nominal = thresholds_.nominalFeatureVectorPtr_->getValue(featureIndex);
        std::unique_ptr<IHeadRefinement> headRefinementPtr =
            thresholds_.headRefinementFactoryPtr_->create(labelIndices);
        std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex);
        return std::make_unique<ExactRuleRefinement<T>>(std::move(headRefinementPtr), labelIndices, weights_,
                                                        sumOfWeights_, featureIndex, nominal, std::move(callbackPtr));
    }

    public:

        /**
         * @param thresholds    A reference to an object of type `ExactThresholds` that stores the thresholds
         * @param weights       A reference to an object of type `IWeightVector` that provides access to the weights of
         *                      the individual training examples
         */
        ThresholdsSubset(ExactThresholds& thresholds, const IWeightVector& weights)
            : thresholds_(thresholds), weights_(weights), coverageMask_(CoverageMask(thresholds.getNumExamples())) {
            sumOfWeights_ = weights.getSumOfWeights();
            numModifications_ = 0;
        }

        std::unique_ptr<IRuleRefinement> createRuleRefinement(const FullIndexVector& labelIndices,
                                                              uint32 featureIndex) override {
            return createExactRuleRefinement(labelIndices, featureIndex);
        }

        std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                              uint32 featureIndex) override {
            return createExactRuleRefinement(labelIndices, featureIndex);
        }

        void filterThresholds(Refinement& refinement) override {
            numModifications_++;
            sumOfWeights_ = refinement.coveredWeights;

            uint32 featureIndex = refinement.featureIndex;
            auto cacheFilteredIterator = cacheFiltered_.find(featureIndex);
            FilteredCacheEntry<FeatureVector>& cacheEntry = cacheFilteredIterator->second;
            FeatureVector* featureVector = cacheEntry.vectorPtr.get();

            if (featureVector == nullptr) {
                auto cacheIterator = thresholds_.cache_.find(featureIndex);
                featureVector = cacheIterator->second.get();
            }

            // If there are examples with zero weights, those examples have not been considered considered when
            // searching for the refinement. In the next step, we need to identify the examples that are covered by the
            // refined rule, including those that have previously been ignored, via the function
            // `filterCurrentVector`. Said function calculates the number of covered examples based on the variable
            // `refinement.end`, which represents the position that separates the covered from the uncovered examples.
            // However, when taking into account the examples with zero weights, this position may differ from the
            // current value of `refinement.end` and therefore must be adjusted...
            if (weights_.hasZeroWeights() && abs(refinement.previous - refinement.end) > 1) {
                refinement.end = adjustSplit(*featureVector, refinement.end, refinement.previous, refinement.threshold);
            }

            // Identify the examples that are covered by the refined rule...
            filterCurrentVector<FeatureVector>(cacheEntry, *featureVector, refinement.start, refinement.end,
                                               refinement.comparator, refinement.covered, numModifications_,
                                               coverageMask_, *thresholds_.statisticsPtr_, weights_);
        }

        void filterThresholds(const Condition& condition) override {
            numModifications_++;
            sumOfWeights_ = condition.coveredWeights;

            uint32 featureIndex = condition.featureIndex;
            auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex,
                                                                FilteredCacheEntry<FeatureVector>()).first;
            FilteredCacheEntry<FeatureVector>& cacheEntry = cacheFilteredIterator->second;
            FeatureVector* featureVector = cacheEntry.vectorPtr.get();

            if (featureVector == nullptr) {
                auto cacheIterator = thresholds_.cache_.emplace(featureIndex, std::unique_ptr<FeatureVector>()).first;
                featureVector = cacheIterator->second.get();
            }

            // Identify the examples that are covered by the refined rule...
            filterCurrentVector<FeatureVector>(cacheEntry, *featureVector, condition.start, condition.end,
                                               condition.comparator, condition.covered, numModifications_,
                                               coverageMask_, *thresholds_.statisticsPtr_, weights_);
        }

        void resetThresholds() override {
            numModifications_ = 0;
            sumOfWeights_ = weights_.getSumOfWeights();
            cacheFiltered_.clear();
            coverageMask_.reset();
        }

        const CoverageMask& getCoverageMask() const {
            return coverageMask_;
        }

        float64 evaluateOutOfSample(const CoverageMask& coverageMask, const AbstractPrediction& head) const override {
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createSubset(*thresholds_.statisticsPtr_);
            uint32 numExamples = thresholds_.getNumExamples();

            for (uint32 r = 0; r < numExamples; r++) {
                if (weights_.getValue(r) == 0 && coverageMask.isCovered(r)) {
                    statisticsSubsetPtr->addToSubset(r, 1);
                }
            }

            std::unique_ptr<IHeadRefinement> headRefinementPtr = head.createHeadRefinement(
                *thresholds_.headRefinementFactoryPtr_);
            const EvaluatedPrediction& prediction = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false,
                                                                                           false);
            return prediction.overallQualityScore;
        }

        void recalculatePrediction(const CoverageMask& coverageMask, Refinement& refinement) const override {
            AbstractPrediction& head = *refinement.headPtr;
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createSubset(*thresholds_.statisticsPtr_);
            uint32 numExamples = thresholds_.getNumExamples();

            for (uint32 r = 0; r < numExamples; r++) {
                if (coverageMask.isCovered(r)) {
                    statisticsSubsetPtr->addToSubset(r, 1);
                }
            }

            std::unique_ptr<IHeadRefinement> headRefinementPtr = head.createHeadRefinement(
                *thresholds_.headRefinementFactoryPtr_);
            const EvaluatedPrediction& prediction = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false,
                                                                                           false);
            const EvaluatedPrediction::const_iterator updatedIterator = prediction.cbegin();
            AbstractPrediction::iterator iterator = head.begin();
            uint32 numElements = head.getNumElements();

            for (uint32 c = 0; c < numElements; c++) {
                iterator[c] = updatedIterator[c];
            }
        }

        void applyPrediction(const AbstractPrediction& prediction) override {
            uint32 numExamples = thresholds_.getNumExamples();

            for (uint32 r = 0; r < numExamples; r++) {
                if (coverageMask_.isCovered(r)) {
                    prediction.apply(*thresholds_.statisticsPtr_, r);
                }
            }
        }

};

ExactThresholds::ExactThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                 std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                 std::shared_ptr<AbstractStatistics> statisticsPtr,
                                 std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr)
    : AbstractThresholds(featureMatrixPtr, nominalFeatureVectorPtr, statisticsPtr, headRefinementFactoryPtr) {

}

std::unique_ptr<IThresholdsSubset> ExactThresholds::createSubset(const IWeightVector& weights) {
    updateSampledStatistics(*statisticsPtr_, weights);
    return std::make_unique<ExactThresholds::ThresholdsSubset>(*this, weights);
}
