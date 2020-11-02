#include "thresholds.h"
#include <cstddef>
#include <stdlib.h>


/**
 * Updates the given statistics by applying the weights of the individual training examples.
 *
 * @param statistics    A reference to an object of type `AbstractStatistics` to be updated
 * @param weights       A reference to an object of type `IWeightVector` that provides access to the weights of the
 *                      individual training examples
 */
static inline void updateSampledStatistics(AbstractStatistics& statistics, const IWeightVector& weights) {
    uint32 numExamples = statistics.getNumRows();
    statistics.resetSampledStatistics();

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 weight = weights.getValue(r);
        statistics.addSampledStatistic(r, weight);
    }
}

/**
 * Adjusts the position that separates the examples that are covered by a condition from the ones that are not covered,
 * with respect to those examples that are not contained in the current sub-sample. This requires to look back a certain
 * number of examples to see if they satisfy the new condition or not. I.e., to traverse the examples in ascending or
 * descending order, depending on whether `conditionEnd` is smaller than `conditionPrevious` or vice versa, until the
 * next example that is contained in the current sub-sampling is encountered.
 *
 * @param featureVector     A reference to an object of type `FeatureVector` that stores the indices and feature values
 *                          of the training examples
 * @param conditionEnd      The position that separates the covered from the uncovered examples when only taking into
 *                          account the examples that are contained in the current sub-sample
 * @param conditionPrevious The position to stop at (exclusive)
 * @param threshold         The threshold of the condition
 * @return                  The adjusted position that separates the covered from the uncovered examples with respect to
 *                          the examples that are not contained in the current sub-sample
 */
static inline intp adjustSplit(FeatureVector& featureVector, intp conditionEnd, intp conditionPrevious,
                               float32 threshold) {
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    intp adjustedPosition = conditionEnd;
    bool ascending = conditionEnd < conditionPrevious;
    intp direction = ascending ? 1 : -1;
    intp start = conditionEnd + direction;
    uint32 numSteps = abs(start - conditionPrevious);

    // Traverse the examples in ascending (or descending) order until we encounter an example that is contained in the
    // current sub-sample...
    for (uint32 i = 0; i < numSteps; i++) {
        // Check if the current position should be adjusted, or not. This is the case, if the feature value of the
        // current example is smaller than or equal to the given `threshold` (or greater than the `threshold`, if we
        // traverse in descending direction)
        uint32 r = start + (i * direction);
        float32 featureValue = iterator[r].value;
        bool adjust = ascending ? featureValue <= threshold : featureValue > threshold;

        if (adjust) {
            // Update the adjusted position and continue...
            adjustedPosition = r;
        } else {
            // If we have found the first example that is separated from the example at the position we started at, we
            // are done...
            break;
        }
    }

    return adjustedPosition;
}

/**
 * Filters a feature vector that contains the indices of the examples that are covered by the previous rule, as well as
 * their values for a certain feature, after a new condition that corresponds to said feature has been added, such that
 * the filtered vector does only contain the indices and feature values of the examples that are covered by the new
 * rule. The filtered vector is stored in a given struct of type `CacheEntry` and the given statistics are updated
 * accordingly.
 *
 * @param cacheEntry            A reference to a struct of type `CacheEntry` that should be used to store the filtered
 *                              feature vector
 * @param featureVector         A reference to an object of type `FeatureVector` that should be filtered
 * @param conditionStart        The element in `featureVector` that corresponds to the first example (inclusive)
 *                              included in the `IStatisticsSubset` that is covered by the new condition
 * @param conditionEnd          The element in `featureVector` that corresponds to the last example (exclusive)
 * @param conditionComparator   The type of the operator that is used by the new condition
 * @param covered               True, if the examples in range [conditionStart, conditionEnd) are covered by the new
 *                              condition and the remaining ones are not, false, if the examples in said range are not
 *                              covered and the remaining ones are
 * @param numConditions         The total number of conditions in the rule's body (including the new one)
 * @param coveredExamplesMask   An array of type `uint32`, shape `(num_examples)` that is used to keep track of the
 *                              indices of the examples that are covered by the previous rule. It will be updated by
 *                              this function
 * @param coveredExamplesTarget The value that is used to mark those elements in `coveredExamplesMask` that are covered
 *                              by the previous rule
 * @param statistics            A reference to an object of type `AbstractStatistics` to be notified about the examples
 *                              that must be considered when searching for the next refinement, i.e., the examples that
 *                              are covered by the new rule
 * @param weights               A reference to an an object of type `IWeightVector` that provides access to the weights
 *                              of the training examples
 * @return                      The value that is used to mark those elements in the updated `coveredExamplesMask` that
 *                              are covered by the new rule
 */
static inline uint32 filterCurrentFeatureVector(CacheEntry& cacheEntry, FeatureVector& featureVector,
                                                intp conditionStart, intp conditionEnd, Comparator conditionComparator,
                                                bool covered, uint32 numConditions, uint32* coveredExamplesMask,
                                                uint32 coveredExamplesTarget, AbstractStatistics& statistics,
                                                const IWeightVector& weights) {
    uint32 numTotalElements = featureVector.getNumElements();
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    bool descending = conditionEnd < conditionStart;
    uint32 updatedTarget;

    // Determine the number of elements in the filtered vector...
    uint32 numConditionSteps = abs(conditionStart - conditionEnd);
    uint32 numElements = covered ? numConditionSteps :
        (numTotalElements > numConditionSteps ? numTotalElements - numConditionSteps : 0);

    // Create a new vector that will contain the filtered elements...
    std::unique_ptr<FeatureVector> filteredVectorPtr = std::make_unique<FeatureVector>(numElements);
    FeatureVector::iterator filteredIterator = filteredVectorPtr->begin();

    intp direction;
    uint32 i;

    if (descending) {
        direction = -1;
        i = numElements - 1;
    } else {
        direction = 1;
        i = 0;
    }

    if (covered) {
        updatedTarget = numConditions;
        statistics.resetCoveredStatistics();

        // Retain the indices at positions [conditionStart, conditionEnd) and set the corresponding values in
        // `coveredExamplesMasK` to `numConditions`, which marks them as covered (because
        // `updatedTarget == numConditions`)...
        for (uint32 j = 0; j < numConditionSteps; j++) {
            uint32 r = conditionStart + (j * direction);
            uint32 index = iterator[r].index;
            coveredExamplesMask[index] = numConditions;
            filteredIterator[i].index = index;
            filteredIterator[i].value = iterator[r].value;
            uint32 weight = weights.getValue(index);
            statistics.updateCoveredStatistic(index, weight, false);
            i += direction;
        }
    } else {
        updatedTarget = coveredExamplesTarget;
        intp start, end;

        if (descending) {
            start = numTotalElements - 1;
            end = -1;
        } else {
            start = 0;
            end = numTotalElements;
        }

        if (conditionComparator == NEQ) {
            // Retain the indices at positions [start, conditionStart), while leaving the corresponding values in
            // `coveredExamplesMask` untouched, such that all previously covered examples in said range are still marked
            // as covered, while previously uncovered examples are still marked as uncovered...
            uint32 numSteps = abs(start - conditionStart);

            for (uint32 j = 0; j < numSteps; j++) {
                uint32 r = start + (j * direction);
                filteredIterator[i].index = iterator[r].index;
                filteredIterator[i].value = iterator[r].value;
                i += direction;
            }
        }

        // Discard the indices at positions [conditionStart, conditionEnd) and set the corresponding values in
        // `coveredExamplesMask` to `numConditions`, which marks them as uncovered (because
        // `updatedTarget != numConditions`)...
        for (uint32 j = 0; j < numConditionSteps; j++) {
            uint32 r = conditionStart + (j * direction);
            uint32 index = iterator[r].index;
            coveredExamplesMask[index] = numConditions;
            uint32 weight = weights.getValue(index);
            statistics.updateCoveredStatistic(index, weight, true);
        }

        // Retain the indices at positions [conditionEnd, end), while leaving the corresponding values in
        // `coveredExamplesMask` untouched, such that all previously covered examples in said range are still marked as
        // covered, while previously uncovered examples are still marked as uncovered...
        uint32 numSteps = abs(conditionEnd - end);

        for (uint32 j = 0; j < numSteps; j++) {
            uint32 r = conditionEnd + (j * direction);
            filteredIterator[i].index = iterator[r].index;
            filteredIterator[i].value = iterator[r].value;
            i += direction;
        }
    }

    cacheEntry.featureVectorPtr = std::move(filteredVectorPtr);
    cacheEntry.numConditions = numConditions;
    return updatedTarget;
}

/**
 * Filters a feature vector that contains the indices of training examples, as well as their values for a certain
 * feature, such that the filtered vector does only contain the indices and feature values of those examples that are
 * covered by the current rule. The filtered vector is stored in a given struct of type `CacheEntry`.
 *
 * @param indexedArray          A reference to an object of type `FeatureVector` that should be filtered
 * @param cacheEntry            A reference to a struct of type `CacheEntry` that should be used to store the filtered
 *                              vector
 * @param numConditions         The total number of conditions in the current rule's body
 * @param coveredExamplesMask   An array of type `uint32`, shape `(num_examples)`, that is used to keep track of the
 *                              indices of the examples that are covered by the current rule
 * @param coveredExamplesTarget The value that is used to mark those elements in `coveredExamplesMask` that are covered
 *                              by the current rule
 */
static inline void filterAnyFeatureVector(FeatureVector& featureVector, CacheEntry& cacheEntry, uint32 numConditions,
                                          const uint32* coveredExamplesMask, uint32 coveredExamplesTarget) {
    uint32 maxElements = featureVector.getNumElements();
    FeatureVector* filteredVector = cacheEntry.featureVectorPtr.get();

    if (filteredVector == nullptr) {
        cacheEntry.featureVectorPtr = std::move(std::make_unique<FeatureVector>(maxElements));
        filteredVector = cacheEntry.featureVectorPtr.get();
    }

    FeatureVector::const_iterator iterator = featureVector.cbegin();
    FeatureVector::iterator filteredIterator = filteredVector->begin();
    uint32 i = 0;

    for (uint32 r = 0; r < maxElements; r++) {
        uint32 index = iterator[r].index;

        if (coveredExamplesMask[index] == coveredExamplesTarget) {
            filteredIterator[i].index = index;
            filteredIterator[i].value = iterator[r].value;
            i++;
        }
    }

    filteredVector->setNumElements(i);
    cacheEntry.numConditions = numConditions;
}

/**
 * Provides access to a subset of the thresholds that are stored by an instance of the class `ExactThresholdsImpl`.
 */
class ExactThresholdsImpl::ThresholdsSubset : virtual public IThresholdsSubset {

    private:

    /**
     * A callback that allows to retrieve feature vectors. If available, the feature vectors are retrieved from the
     * cache. Otherwise, they are fetched from the feature matrix.
     */
    class Callback : virtual public IRuleRefinementCallback<FeatureVector> {

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
                CacheEntry& cacheEntry = cacheFilteredIterator->second;
                FeatureVector* featureVector = cacheEntry.featureVectorPtr.get();

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
                uint32 numConditions = thresholdsSubset_.numRefinements_;

                if (numConditions > cacheEntry.numConditions) {
                    filterAnyFeatureVector(*featureVector, cacheEntry, numConditions,
                                           thresholdsSubset_.coveredExamplesMask_,
                                           thresholdsSubset_.coveredExamplesTarget_);
                    featureVector = cacheEntry.featureVectorPtr.get();
                }

                return std::make_unique<Result>(*thresholdsSubset_.thresholds_.statisticsPtr_, *featureVector);
            }

    };

    ExactThresholdsImpl& thresholds_;

    const IWeightVector& weights_;

    uint32 sumOfWeights_;

    uint32* coveredExamplesMask_;

    uint32 coveredExamplesTarget_;

    uint32 numRefinements_;

    std::unordered_map<uint32, CacheEntry> cacheFiltered_;

    template<class T>
    std::unique_ptr<IRuleRefinement> createExactRuleRefinement(const T& labelIndices, uint32 featureIndex) {
        // Retrieve the `CacheEntry` from the cache, or insert a new one if it does not already exist...
        auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, CacheEntry()).first;
        FeatureVector* featureVector = cacheFilteredIterator->second.featureVectorPtr.get();

        // If the `CacheEntry` in the cache does not refer to a `FeatureVector`, add an empty `unique_ptr` to the
        // cache...
        if (featureVector == nullptr) {
            thresholds_.cache_.emplace(featureIndex, std::unique_ptr<FeatureVector>());
        }

        bool nominal = thresholds_.nominalFeatureVectorPtr_->getValue(featureIndex);
        std::unique_ptr<IHeadRefinement> headRefinementPtr =
            thresholds_.headRefinementFactoryPtr_->create(labelIndices);
        std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex);
        return std::make_unique<ExactRuleRefinementImpl<T>>(std::move(headRefinementPtr), labelIndices, weights_,
                                                            sumOfWeights_, featureIndex, nominal,
                                                            std::move(callbackPtr));
    }

    public:

        /**
         * @param thresholds    A reference to an object of type `ExactThresholdsImpl` that stores the thresholds
         * @param weights       A reference to an object of type `IWeightVector` that provides access to the weights of
         *                      the individual training examples
         */
        ThresholdsSubset(ExactThresholdsImpl& thresholds, const IWeightVector& weights)
            : thresholds_(thresholds), weights_(weights) {
            sumOfWeights_ = weights.getSumOfWeights();
            uint32 numExamples = thresholds.getNumRows();
            coveredExamplesMask_ = new uint32[numExamples]{0};
            coveredExamplesTarget_ = 0;
            numRefinements_ = 0;
        }

        ~ThresholdsSubset() {
            delete[] coveredExamplesMask_;
        }

        std::unique_ptr<IRuleRefinement> createRuleRefinement(const FullIndexVector& labelIndices,
                                                              uint32 featureIndex) override {
            return createExactRuleRefinement(labelIndices, featureIndex);
        }

        std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                              uint32 featureIndex) override {
            return createExactRuleRefinement(labelIndices, featureIndex);
        }

        void applyRefinement(Refinement& refinement) override {
            numRefinements_++;
            sumOfWeights_ = refinement.coveredWeights;

            uint32 featureIndex = refinement.featureIndex;
            auto cacheFilteredIterator = cacheFiltered_.find(featureIndex);
            CacheEntry& cacheEntry = cacheFilteredIterator->second;
            FeatureVector* featureVector = cacheEntry.featureVectorPtr.get();

            if (featureVector == nullptr) {
                auto cacheIterator = thresholds_.cache_.find(featureIndex);
                featureVector = cacheIterator->second.get();
            }

            // If there are examples with zero weights, those examples have not been considered considered when
            // searching for the refinement. In the next step, we need to identify the examples that are covered by the
            // refined rule, including those that have previously been ignored, via the function
            // `filterCurrentFeatureVector`. Said function calculates the number of covered examples based on the
            // variable `refinement.end`, which represents the position that separates the covered from the uncovered
            // examples. However, when taking into account the examples with zero weights, this position may differ from
            // the current value of `refinement.end` and therefore must be adjusted...
            if (weights_.hasZeroWeights() && abs(refinement.previous - refinement.end) > 1) {
                refinement.end = adjustSplit(*featureVector, refinement.end, refinement.previous, refinement.threshold);
            }

            // Identify the examples that are covered by the refined rule...
            coveredExamplesTarget_ = filterCurrentFeatureVector(cacheEntry, *featureVector, refinement.start,
                                                                refinement.end, refinement.comparator,
                                                                refinement.covered, numRefinements_,
                                                                coveredExamplesMask_, coveredExamplesTarget_,
                                                                *thresholds_.statisticsPtr_, weights_);
        }

        void recalculatePrediction(Refinement& refinement) const override {
            AbstractPrediction& head = *refinement.headPtr;
            std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createSubset(*thresholds_.statisticsPtr_);
            uint32 numExamples = thresholds_.getNumRows();

            for (uint32 r = 0; r < numExamples; r++) {
                if (coveredExamplesMask_[r] == coveredExamplesTarget_) {
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
            uint32 numExamples = thresholds_.getNumRows();

            for (uint32 r = 0; r < numExamples; r++) {
                if (coveredExamplesMask_[r] == coveredExamplesTarget_) {
                    prediction.apply(*thresholds_.statisticsPtr_, r);
                }
            }
        }

};

AbstractThresholds::AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                       std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                       std::shared_ptr<AbstractStatistics> statisticsPtr,
                                       std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr)
    : featureMatrixPtr_(featureMatrixPtr), nominalFeatureVectorPtr_(nominalFeatureVectorPtr),
      statisticsPtr_(statisticsPtr), headRefinementFactoryPtr_(headRefinementFactoryPtr) {

}

uint32 AbstractThresholds::getNumRows() const {
    return featureMatrixPtr_->getNumRows();
}

uint32 AbstractThresholds::getNumCols() const {
    return featureMatrixPtr_->getNumCols();
}

uint32 AbstractThresholds::getNumLabels() const {
    return statisticsPtr_->getNumCols();
}

ExactThresholdsImpl::ExactThresholdsImpl(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                         std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                         std::shared_ptr<AbstractStatistics> statisticsPtr,
                                         std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr)
    : AbstractThresholds(featureMatrixPtr, nominalFeatureVectorPtr, statisticsPtr, headRefinementFactoryPtr) {

}

std::unique_ptr<IThresholdsSubset> ExactThresholdsImpl::createSubset(const IWeightVector& weights) {
    updateSampledStatistics(*statisticsPtr_, weights);
    return std::make_unique<ExactThresholdsImpl::ThresholdsSubset>(*this, weights);
}

/**
 * Provides access to a subset of the thresholds that are stored by an instance of the class
 * `ApproximateThresholdsImpl`.
 */
class ApproximateThresholdsImpl::ThresholdsSubset : virtual public IThresholdsSubset {

    private:

        /**
         * A callback that allows to retrieve bins and corresponding statistics. If available, the bins and statistics
         * are retrieved from the cache. Otherwise, they are computed by fetching the feature values from the feature
         * matrix and applying a binning method.
         */
        class Callback : virtual public IBinningObserver, virtual public IRuleRefinementCallback<BinVector> {

            private:

                ThresholdsSubset& thresholdsSubset_;

                uint32 featureIndex_;

                std::unique_ptr<AbstractStatistics::IHistogramBuilder> histogramBuilderPtr_;

                BinVector* currentBinVector_;

            public:

                /**
                 * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the bins
                 * @param featureIndex      The index of the feature for which the bins should be retrieved
                 */
                Callback(ThresholdsSubset& thresholdsSubset, uint32 featureIndex)
                    : thresholdsSubset_(thresholdsSubset), featureIndex_(featureIndex) {

                }

                std::unique_ptr<Result> get() override {
                    auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                    BinCacheEntry& binCacheEntry = cacheIterator->second;

                    if (binCacheEntry.binVectorPtr.get() == nullptr) {
                        std::unique_ptr<FeatureVector> featureVectorPtr;
                        thresholdsSubset_.thresholds_.featureMatrixPtr_->fetchFeatureVector(featureIndex_, featureVectorPtr);
                        uint32 numBins = thresholdsSubset_.thresholds_.binningPtr_->getNumBins(*featureVectorPtr);
                        binCacheEntry.binVectorPtr =  std::move(std::make_unique<BinVector>(numBins, true));
                        histogramBuilderPtr_ = thresholdsSubset_.thresholds_.statisticsPtr_->buildHistogram(numBins);
                        currentBinVector_ = binCacheEntry.binVectorPtr.get();
                        thresholdsSubset_.thresholds_.binningPtr_->createBins(numBins, *featureVectorPtr, *this);
                        binCacheEntry.statisticsPtr = std::move(histogramBuilderPtr_->build());
                    }

                    return std::make_unique<Result>(*binCacheEntry.statisticsPtr, *binCacheEntry.binVectorPtr);
                }

                void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override {
                    BinVector::iterator binIterator = currentBinVector_->begin();
                    binIterator[binIndex].numExamples += 1;
                    float32 currentValue = entry.value;

                    if (currentValue < binIterator[binIndex].minValue) {
                        binIterator[binIndex].minValue = currentValue;
                    }

                    if (binIterator[binIndex].maxValue < currentValue) {
                        binIterator[binIndex].maxValue = currentValue;
                    }

                    histogramBuilderPtr_->onBinUpdate(binIndex, entry);
                }

        };

        ApproximateThresholdsImpl& thresholds_;

        template<class T>
        std::unique_ptr<IRuleRefinement> createApproximateRuleRefinement(const T& labelIndices, uint32 featureIndex) {
            thresholds_.cache_.emplace(featureIndex, BinCacheEntry());
            std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex);
            std::unique_ptr<IHeadRefinement> headRefinementPtr =
                thresholds_.headRefinementFactoryPtr_->create(labelIndices);
            return std::make_unique<ApproximateRuleRefinementImpl<T>>(std::move(headRefinementPtr), labelIndices,
                                                                      featureIndex, std::move(callbackPtr));
        }

    public:

        /**
         * @param thresholds A reference to an object of type `ApproximateThresholdsImpl` that stores the thresholds
         */
        ThresholdsSubset(ApproximateThresholdsImpl& thresholds)
            : thresholds_(thresholds) {

        }

        std::unique_ptr<IRuleRefinement> createRuleRefinement(const FullIndexVector& labelIndices,
                                                              uint32 featureIndex) override {
            return createApproximateRuleRefinement(labelIndices, featureIndex);
        }

        std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                              uint32 featureIndex) override {
            return createApproximateRuleRefinement(labelIndices, featureIndex);
        }

        void applyRefinement(Refinement& refinement) override {

        }

        void recalculatePrediction(Refinement& refinement) const override {

        }

        void applyPrediction(const AbstractPrediction& prediction) override {

        }

};

ApproximateThresholdsImpl::ApproximateThresholdsImpl(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                                     std::shared_ptr<INominalFeatureVector> nominalFeatureVectorPtr,
                                                     std::shared_ptr<AbstractStatistics> statisticsPtr,
                                                     std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                                                     std::shared_ptr<IBinning> binningPtr)
    : AbstractThresholds(featureMatrixPtr, nominalFeatureVectorPtr, statisticsPtr, headRefinementFactoryPtr),
      binningPtr_(binningPtr) {

}

std::unique_ptr<IThresholdsSubset> ApproximateThresholdsImpl::createSubset(const IWeightVector& weights) {
    updateSampledStatistics(*statisticsPtr_, weights);
    return std::make_unique<ApproximateThresholdsImpl::ThresholdsSubset>(*this);
}
