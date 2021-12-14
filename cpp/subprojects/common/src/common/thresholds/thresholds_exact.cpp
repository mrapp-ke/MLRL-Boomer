#include "common/thresholds/thresholds_exact.hpp"
#include "common/rule_refinement/rule_refinement_exact.hpp"
#include "common/validation.hpp"
#include "thresholds_common.hpp"
#include <unordered_map>
#include <cmath>


/**
 * An entry that is stored in a cache and contains an unique pointer to a feature vector. The field `numConditions`
 * specifies how many conditions the rule contained when the vector was updated for the last time. It may be used to
 * check if the vector is still valid or must be updated.
 */
struct FilteredCacheEntry {

    FilteredCacheEntry(): numConditions(0) { };

    /**
     * An unique pointer to an object of type `FeatureVector` that stores feature values.
     */
    std::unique_ptr<FeatureVector> vectorPtr;

    /**
     * The number of conditions that were contained by the rule when the cache was updated for the last time.
     */
    uint32 numConditions;

};

static inline uint32 adjustSplit(FeatureVector::const_iterator iterator, uint32 start, uint32 end, float32 threshold) {
    while (start <= end) {
        uint32 pivot = start + ((end - start) / 2);
        float32 featureValue = iterator[pivot].value;

        if (featureValue <= threshold) {
            start = pivot + 1;
        } else {
            end = pivot - 1;
        }
    }

    return start;
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
static inline int64 adjustSplit(FeatureVector& featureVector, int64 conditionEnd, int64 conditionPrevious,
                                float32 threshold) {
    FeatureVector::const_iterator iterator = featureVector.cbegin();

    if (conditionEnd < conditionPrevious) {
        return adjustSplit(iterator, conditionEnd + 1, conditionPrevious - 1, threshold) - 1;
    } else {
        return adjustSplit(iterator, conditionPrevious + 1, conditionEnd - 1, threshold);
    }
}

/**
 * Filters a given feature vector, which contains the elements for a certain feature that are covered by the previous
 * rule, after a new condition that corresponds to said feature has been added, such that the filtered vector does only
 * contain the elements that are covered by the new rule. The filtered vector is stored in a given struct of type
 * `FilteredCacheEntry` and the given statistics are updated accordingly.
 *
 * @param vector                A reference to an object of type `FeatureVector` that should be filtered
 * @param cacheEntry            A reference to a struct of type `FilteredCacheEntry` that should be used to store the
 *                              filtered feature vector
 * @param conditionStart        The element in `vector` that corresponds to the first statistic (inclusive) included in
 *                              the `IStatisticsSubset` that is covered by the new condition
 * @param conditionEnd          The element in `vector` that corresponds to the last statistic (exclusive)
 * @param conditionComparator   The type of the operator that is used by the new condition
 * @param covered               True, if the elements in range [conditionStart, conditionEnd) are covered by the new
 *                              condition and the remaining ones are not, false, if the elements in said range are not
 *                              covered, but the remaining ones are
 * @param numConditions         The total number of conditions in the rule's body (including the new one)
 * @param coverageMask          A reference to an object of type `CoverageMask` that is used to keep track of the
 *                              elements that are covered by the previous rule. It will be updated by this function
 * @param statistics            A reference to an object of type `IStatistics` to be notified about the statistics that
 *                              must be considered when searching for the next refinement, i.e., the statistics that are
 *                              covered by the new rule
 * @param weights               A reference to an an object of type `IWeightVector` that provides access to the weights
 *                              of the individual training examples
 */
static inline void filterCurrentVector(const FeatureVector& vector, FilteredCacheEntry& cacheEntry,
                                       int64 conditionStart, int64 conditionEnd, Comparator conditionComparator,
                                       bool covered, uint32 numConditions, CoverageMask& coverageMask,
                                       IStatistics& statistics, const IWeightVector& weights) {
    // Determine the number of elements in the filtered vector...
    uint32 numTotalElements = vector.getNumElements();
    uint32 distance = std::abs(conditionStart - conditionEnd);
    uint32 numElements = covered ? distance : (numTotalElements > distance ? numTotalElements - distance : 0);

    // Create a new vector that will contain the filtered elements, if necessary...
    FeatureVector* filteredVector = cacheEntry.vectorPtr.get();

    if (filteredVector == nullptr) {
        cacheEntry.vectorPtr = std::make_unique<FeatureVector>(numElements);
        filteredVector = cacheEntry.vectorPtr.get();
    }

    typename FeatureVector::const_iterator iterator = vector.cbegin();
    FeatureVector::iterator filteredIterator = filteredVector->begin();
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    bool descending = conditionEnd < conditionStart;
    int64 start, end;

    if (descending) {
        start = conditionEnd + 1;
        end = conditionStart + 1;
    } else {
        start = conditionStart;
        end = conditionEnd;
    }

    if (covered) {
        coverageMask.setIndicatorValue(numConditions);
        statistics.resetCoveredStatistics();
        uint32 i = 0;

        // Retain the indices at positions [start, end) and set the corresponding values in the given `coverageMask` to
        // `numConditions` to mark them as covered...
        for (int64 r = start; r < end; r++) {
            uint32 index = iterator[r].index;
            coverageMaskIterator[index] = numConditions;
            filteredIterator[i].index = index;
            filteredIterator[i].value = iterator[r].value;
            float64 weight = weights.getWeight(index);
            statistics.updateCoveredStatistic(index, weight, false);
            i++;
        }
    } else {
        // Discard the indices at positions [start, end) and set the corresponding values in `coverageMask` to
        // `numConditions`, which marks them as uncovered...
        for (int64 r = start; r < end; r++) {
            uint32 index = iterator[r].index;
            coverageMaskIterator[index] = numConditions;
            float64 weight = weights.getWeight(index);
            statistics.updateCoveredStatistic(index, weight, true);
        }

        if (conditionComparator == NEQ) {
            // Retain the indices at positions [currentStart, currentEnd), while leaving the corresponding values in
            // `coverageMask` untouched, such that all previously covered examples in said range are still marked
            // as covered, while previously uncovered examples are still marked as uncovered...
            int64 currentStart, currentEnd;
            uint32 i;

            if (descending) {
                currentStart = end;
                currentEnd = numTotalElements;
                i = start;
            } else {
                currentStart = 0;
                currentEnd = start;
                i = 0;
            }

            for (int64 r = currentStart; r < currentEnd; r++) {
                filteredIterator[i].index = iterator[r].index;
                filteredIterator[i].value = iterator[r].value;
                i++;
            }
        }

        // Retain the indices at positions [currentStart, currentEnd), while leaving the corresponding values in
        // `coverageMask` untouched, such that all previously covered examples in said range are still marked as
        // covered, while previously uncovered examples are still marked as uncovered...
        int64 currentStart, currentEnd;
        uint32 i;

        if (descending) {
            currentStart = 0;
            currentEnd = start;
            i = 0;
        } else {
            currentStart = end;
            currentEnd = numTotalElements;
            i = start;
        }

        for (int64 r = currentStart; r < currentEnd; r++) {
            filteredIterator[i].index = iterator[r].index;
            filteredIterator[i].value = iterator[r].value;
            i++;
        }

        // Iterate the indices of examples with missing feature values and set the corresponding values in
        // `coverageMask` to `numConditions`, which marks them as uncovered...
        for (auto it = vector.missing_indices_cbegin(); it != vector.missing_indices_cend(); it++) {
            uint32 index = *it;
            coverageMaskIterator[index] = numConditions;
            float64 weight = weights.getWeight(index);
            statistics.updateCoveredStatistic(index, weight, true);
        }
    }

    filteredVector->setNumElements(numElements, true);
    cacheEntry.numConditions = numConditions;
}

/**
 * Filters a given feature vector, such that the filtered vector does only contain the elements that are covered by the
 * current rule. The filtered vector is stored in a given struct of type `FilteredCacheEntry`.
 *
 * @param vector        A reference to an object of type `FeatureVector` that should be filtered
 * @param cacheEntry    A reference to a struct of type `FilteredCacheEntry` that should be used to store the filtered
 *                      vector
 * @param numConditions The total number of conditions in the current rule's body
 * @param coverageMask  A reference to an object of type `CoverageMask` that is used to keep track of the elements that
 *                      are covered by the current rule
 */
static inline void filterAnyVector(const FeatureVector& vector, FilteredCacheEntry& cacheEntry, uint32 numConditions,
                                   const CoverageMask& coverageMask) {
    uint32 maxElements = vector.getNumElements();
    FeatureVector* filteredVector = cacheEntry.vectorPtr.get();

    if (filteredVector == nullptr) {
        cacheEntry.vectorPtr = std::make_unique<FeatureVector>(maxElements);
        filteredVector = cacheEntry.vectorPtr.get();
    } else {
        filteredVector->clearMissingIndices();
    }

    // Filter the missing indices...
    for (auto it = vector.missing_indices_cbegin(); it != vector.missing_indices_cend(); it++) {
        uint32 index = *it;

        if (coverageMask.isCovered(index)) {
            filteredVector->addMissingIndex(index);
        }
    }

    // Filter the feature values...
    typename FeatureVector::const_iterator iterator = vector.cbegin();
    typename FeatureVector::iterator filteredIterator = filteredVector->begin();
    uint32 i = 0;

    for (uint32 r = 0; r < maxElements; r++) {
        uint32 index = iterator[r].index;

        if (coverageMask.isCovered(index)) {
            filteredIterator[i].index = index;
            filteredIterator[i].value = iterator[r].value;
            i++;
        }
    }

    filteredVector->setNumElements(i, true);
    cacheEntry.numConditions = numConditions;
}

/**
 * Provides access to all thresholds that result from the feature values of the training examples.
 */
class ExactThresholds final : public AbstractThresholds {

    private:

        /**
         * Provides access to a subset of the thresholds that are stored by an instance of the class `ExactThresholds`.
         */
        class ThresholdsSubset final : public IThresholdsSubset {

            private:

            /**
             * A callback that allows to retrieve feature vectors. If available, the feature vectors are retrieved from
             * the cache. Otherwise, they are fetched from the feature matrix.
             */
            class Callback final : public IRuleRefinementCallback<FeatureVector, IWeightVector> {

                private:

                    ThresholdsSubset& thresholdsSubset_;

                    uint32 featureIndex_;

                public:

                    /**
                     * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the
                     *                          feature vectors
                     * @param featureIndex      The index of the feature for which the feature vector should be
                     *                          retrieved
                     */
                    Callback(ThresholdsSubset& thresholdsSubset, uint32 featureIndex)
                        : thresholdsSubset_(thresholdsSubset), featureIndex_(featureIndex) {

                    }

                    std::unique_ptr<Result> get() override {
                        auto cacheFilteredIterator = thresholdsSubset_.cacheFiltered_.find(featureIndex_);
                        FilteredCacheEntry& cacheEntry = cacheFilteredIterator->second;
                        FeatureVector* featureVector = cacheEntry.vectorPtr.get();

                        if (featureVector == nullptr) {
                            auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                            featureVector = cacheIterator->second.get();

                            if (featureVector == nullptr) {
                                thresholdsSubset_.thresholds_.featureMatrix_.fetchFeatureVector(featureIndex_,
                                                                                                cacheIterator->second);
                                cacheIterator->second->sortByValues();
                                featureVector = cacheIterator->second.get();
                            }
                        }

                        // Filter feature vector, if only a subset of its elements are covered by the current rule...
                        uint32 numConditions = thresholdsSubset_.numModifications_;

                        if (numConditions > cacheEntry.numConditions) {
                            filterAnyVector(*featureVector, cacheEntry, numConditions, thresholdsSubset_.coverageMask_);
                            featureVector = cacheEntry.vectorPtr.get();
                        }

                        return std::make_unique<Result>(thresholdsSubset_.thresholds_.statisticsProvider_.get(),
                                                        thresholdsSubset_.weights_, *featureVector);
                    }

            };

            ExactThresholds& thresholds_;

            const IWeightVector& weights_;

            uint32 numCoveredExamples_;

            CoverageMask coverageMask_;

            uint32 numModifications_;

            std::unordered_map<uint32, FilteredCacheEntry> cacheFiltered_;

            template<typename T>
            std::unique_ptr<IRuleRefinement> createExactRuleRefinement(const T& labelIndices, uint32 featureIndex) {
                // Retrieve the `FilteredCacheEntry` from the cache, or insert a new one if it does not already exist...
                auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry()).first;
                FeatureVector* featureVector = cacheFilteredIterator->second.vectorPtr.get();

                // If the `FilteredCacheEntry` in the cache does not refer to a `FeatureVector`, add an empty
                // `unique_ptr` to the cache...
                if (featureVector == nullptr) {
                    thresholds_.cache_.emplace(featureIndex, std::unique_ptr<FeatureVector>());
                }

                bool nominal = thresholds_.nominalFeatureMask_.isNominal(featureIndex);
                std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex);
                return std::make_unique<ExactRuleRefinement<T>>(labelIndices, numCoveredExamples_, featureIndex,
                                                                nominal, std::move(callbackPtr));
            }

            public:

                /**
                 * @param thresholds    A reference to an object of type `ExactThresholds` that stores the thresholds
                 * @param weights       A reference to an object of type `IWeightVector` that provides access to the
                 *                      weights of the individual training examples
                 */
                ThresholdsSubset(ExactThresholds& thresholds, const IWeightVector& weights)
                    : thresholds_(thresholds), weights_(weights), numCoveredExamples_(weights.getNumNonZeroWeights()),
                      coverageMask_(CoverageMask(thresholds.getNumExamples())), numModifications_(0) {

                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const CompleteIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createExactRuleRefinement(labelIndices, featureIndex);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createExactRuleRefinement(labelIndices, featureIndex);
                }

                void filterThresholds(Refinement& refinement) override {
                    numModifications_++;
                    numCoveredExamples_ = refinement.numCovered;

                    uint32 featureIndex = refinement.featureIndex;
                    auto cacheFilteredIterator = cacheFiltered_.find(featureIndex);
                    FilteredCacheEntry& cacheEntry = cacheFilteredIterator->second;
                    FeatureVector* featureVector = cacheEntry.vectorPtr.get();

                    if (featureVector == nullptr) {
                        auto cacheIterator = thresholds_.cache_.find(featureIndex);
                        featureVector = cacheIterator->second.get();
                    }

                    // If there are examples with zero weights, those examples have not been considered considered when
                    // searching for the refinement. In the next step, we need to identify the examples that are covered
                    // by the refined rule, including those that have previously been ignored, via the function
                    // `filterCurrentVector`. Said function calculates the number of covered examples based on the
                    // variable `refinement.end`, which represents the position that separates the covered from the
                    // uncovered examples. However, when taking into account the examples with zero weights, this
                    // position may differ from the current value of `refinement.end` and therefore must be adjusted...
                    if (weights_.hasZeroWeights() && std::abs(refinement.previous - refinement.end) > 1) {
                        refinement.end = adjustSplit(*featureVector, refinement.end, refinement.previous,
                                                     refinement.threshold);
                    }

                    // Identify the examples that are covered by the refined rule...
                    filterCurrentVector(*featureVector, cacheEntry, refinement.start, refinement.end,
                                        refinement.comparator, refinement.covered, numModifications_, coverageMask_,
                                        thresholds_.statisticsProvider_.get(), weights_);
                }

                void filterThresholds(const Condition& condition) override {
                    numModifications_++;
                    numCoveredExamples_ = condition.numCovered;

                    uint32 featureIndex = condition.featureIndex;
                    auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry()).first;
                    FilteredCacheEntry& cacheEntry = cacheFilteredIterator->second;
                    FeatureVector* featureVector = cacheEntry.vectorPtr.get();

                    if (featureVector == nullptr) {
                        auto cacheIterator = thresholds_.cache_.emplace(featureIndex,
                                                                        std::unique_ptr<FeatureVector>()).first;
                        featureVector = cacheIterator->second.get();
                    }

                    // Identify the examples that are covered by the condition...
                    if (numModifications_ > cacheEntry.numConditions) {
                        filterAnyVector(*featureVector, cacheEntry, numModifications_, coverageMask_);
                        featureVector = cacheEntry.vectorPtr.get();
                    }

                    filterCurrentVector(*featureVector, cacheEntry, condition.start, condition.end,
                                        condition.comparator, condition.covered, numModifications_, coverageMask_,
                                        thresholds_.statisticsProvider_.get(), weights_);
                }

                void resetThresholds() override {
                    numModifications_ = 0;
                    numCoveredExamples_ = weights_.getNumNonZeroWeights();
                    cacheFiltered_.clear();
                    coverageMask_.reset();
                }

                const ICoverageState& getCoverageState() const override {
                    return coverageMask_;
                }

                float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<SinglePartition::const_iterator>(
                        partition.cbegin(), partition.getNumElements(), weights_, coverageState,
                        thresholds_.statisticsProvider_.get(), head);
                }

                float64 evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<BiPartition::const_iterator>(
                        partition.first_cbegin(), partition.getNumFirst(), weights_, coverageState,
                        thresholds_.statisticsProvider_.get(), head);
                }

                float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, thresholds_.statisticsProvider_.get(),
                                                         head);
                }

                float64 evaluateOutOfSample(BiPartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, partition,
                                                         thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally<SinglePartition::const_iterator>(
                        partition.cbegin(), partition.getNumElements(), coverageState,
                        thresholds_.statisticsProvider_.get(), refinement);
                }

                void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally<BiPartition::const_iterator>(
                        partition.first_cbegin(), partition.getNumFirst(), coverageState,
                        thresholds_.statisticsProvider_.get(), refinement);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally(coverageState, thresholds_.statisticsProvider_.get(), refinement);
                }

                void recalculatePrediction(BiPartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally(coverageState, partition, thresholds_.statisticsProvider_.get(),
                                                    refinement);
                }

                void applyPrediction(const AbstractPrediction& prediction) override {
                    IStatistics& statistics = thresholds_.statisticsProvider_.get();
                    uint32 numStatistics = statistics.getNumStatistics();
                    const CoverageMask* coverageMaskPtr = &coverageMask_;
                    const AbstractPrediction* predictionPtr = &prediction;
                    IStatistics* statisticsPtr = &statistics;
                    uint32 numThreads = thresholds_.numThreads_;

                    #pragma omp parallel for firstprivate(numStatistics) firstprivate(coverageMaskPtr) \
                    firstprivate(predictionPtr) firstprivate(statisticsPtr) schedule(dynamic) num_threads(numThreads)
                    for (int64 i = 0; i < numStatistics; i++) {
                        if (coverageMaskPtr->isCovered(i)) {
                            predictionPtr->apply(*statisticsPtr, i);
                        }
                    }
                }

        };

        uint32 numThreads_;

        std::unordered_map<uint32, std::unique_ptr<FeatureVector>> cache_;

    public:

        /**
         * @param featureMatrix         A reference to an object of type `IFeatureMatrix` that provides access to the
         *                              feature values of the training examples
         * @param nominalFeatureMask    A reference  to an object of type `INominalFeatureMask` that provides access to
         *                              the information whether individual features are nominal or not
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the labels of the training examples
         * @param numThreads            The number of CPU threads to be used to update statistics in parallel
         */
        ExactThresholds(const IFeatureMatrix& featureMatrix, const INominalFeatureMask& nominalFeatureMask,
                        IStatisticsProvider& statisticsProvider, uint32 numThreads)
            : AbstractThresholds(featureMatrix, nominalFeatureMask, statisticsProvider), numThreads_(numThreads) {

        }

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override {
            updateSampledStatisticsInternally(statisticsProvider_.get(), weights);
            return std::make_unique<ExactThresholds::ThresholdsSubset>(*this, weights);
        }

};

ExactThresholdsFactory::ExactThresholdsFactory(uint32 numThreads)
    : numThreads_(numThreads) {
    assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
}

std::unique_ptr<IThresholds> ExactThresholdsFactory::create(
        const IFeatureMatrix& featureMatrix, const INominalFeatureMask& nominalFeatureMask,
        IStatisticsProvider& statisticsProvider) const {
    return std::make_unique<ExactThresholds>(featureMatrix, nominalFeatureMask, statisticsProvider, numThreads_);
}
