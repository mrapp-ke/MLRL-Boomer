#include "thresholds_approximate.h"
#include "thresholds_common.h"
#include "../binning/bin_vector.h"
#include "../rule_refinement/rule_refinement_approximate.h"
#include <unordered_map>
#include <limits>


/**
 * An entry that is stored in the cache and contains unique pointers to a histogram and a vector that stores bins. The
 * field `numConditions` specifies how many conditions the rule contained when the vector was updated for the last time.
 * It may be used to check if the vector and histogram are still valid or must be updated.
 */
struct FilteredBinCacheEntry : public FilteredCacheEntry<BinVector> {
    std::unique_ptr<IHistogram> histogramPtr;
};

static inline void filterCurrentVector(BinVector& vector, FilteredBinCacheEntry& cacheEntry, intp conditionEnd,
                                       bool covered, uint32 numConditions, CoverageMask& coverageMask) {
    uint32 numTotalElements = vector.getNumElements();
    uint32 numElements = covered ? conditionEnd : (numTotalElements > conditionEnd ? numTotalElements - conditionEnd : 0);
    bool wasEmpty = false;

    BinVector* filteredVector = cacheEntry.vectorPtr.get();

    if (filteredVector == nullptr) {
        cacheEntry.vectorPtr = std::make_unique<BinVector>(numElements);
        filteredVector = cacheEntry.vectorPtr.get();
        wasEmpty = true;
    }

    BinVector::bin_const_iterator binIterator = vector.bins_cbegin();
    BinVector::bin_iterator filteredBinIterator = filteredVector->bins_begin();
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    coverageMask.target = numConditions;
    intp start, end;
    uint32 i = 0;

    if (covered) {
        start = 0;
        end = conditionEnd;
    } else {
        start = conditionEnd;
        end = numTotalElements;
    }

    for (intp r = start; r < end; r++) {
        BinVector::ExampleList& examples = vector.getExamples(r);
        BinVector::ExampleList& filteredExamples = filteredVector->getExamples(r);

        for (BinVector::ExampleList::const_iterator it = examples.cbegin(); it != examples.cend(); it++) {
            BinVector::Example example = *it;
            coverageMaskIterator[example.index] = numConditions;

            if (wasEmpty) {
                filteredExamples.push_front(example);
            }
        }

        filteredBinIterator[i].index = binIterator[r].index;
        filteredBinIterator[i].numExamples = binIterator[r].numExamples;
        filteredBinIterator[i].minValue = binIterator[r].minValue;
        filteredBinIterator[i].maxValue = binIterator[r].maxValue;
        i++;
    }

    filteredVector->setNumElements(numElements, true);
    cacheEntry.numConditions = numConditions;
}

static inline void filterAnyVector(BinVector& vector, FilteredBinCacheEntry& cacheEntry, uint32 numConditions,
                                   const CoverageMask& coverageMask) {
    uint32 maxElements = vector.getNumElements();
    BinVector* filteredVector = cacheEntry.vectorPtr.get();
    bool wasEmpty = false;

    if (filteredVector == nullptr) {
        cacheEntry.vectorPtr = std::make_unique<BinVector>(maxElements);
        filteredVector = cacheEntry.vectorPtr.get();
        wasEmpty = true;
    }

    BinVector::bin_const_iterator binIterator = vector.bins_cbegin();
    BinVector::bin_iterator filteredBinIterator = filteredVector->bins_begin();
    uint32 i = 0;

    for(uint32 r = 0; r < maxElements; r++) {
        float32 maxValue = -std::numeric_limits<float32>::infinity();
        float32 minValue = std::numeric_limits<float32>::infinity();
        uint32 numExamples = 0;
        BinVector::ExampleList& examples = vector.getExamples(r);
        BinVector::ExampleList& filteredExamples = filteredVector->getExamples(r);
        BinVector::ExampleList::const_iterator before = examples.cbefore_begin();

        for (BinVector::ExampleList::const_iterator it = examples.cbegin(); it != examples.cend();) {
            BinVector::Example example = *it;
            uint32 index = example.index;

            if (coverageMask.isCovered(index)) {
                float32 value = example.value;

                if (value < minValue) {
                    minValue = value;
                }

                if (maxValue < value) {
                    maxValue = value;
                }

                if (wasEmpty) {
                    filteredExamples.push_front(example);
                }

                numExamples++;
                before = it;
                it++;
            } else if (!wasEmpty) {
                it = filteredExamples.erase_after(before);
            } else {
                it++;
            }
        }

        if (numExamples > 0) {
            filteredBinIterator[i].index = binIterator[r].index;
            filteredBinIterator[i].numExamples = numExamples;
            filteredBinIterator[i].minValue = minValue;
            filteredBinIterator[i].maxValue = maxValue;
            i++;
        }
    }

    filteredVector->setNumElements(i, true);
    cacheEntry.numConditions = numConditions;
}

static inline void buildHistogram(BinVector& vector, const IStatistics& statistics, FilteredBinCacheEntry& cacheEntry) {
    uint32 numBins = vector.getNumElements();
    std::unique_ptr<IStatistics::IHistogramBuilder> histogramBuilderPtr = statistics.createHistogramBuilder(numBins);

    for (uint32 binIndex = 0; binIndex < numBins; binIndex++) {
        BinVector::ExampleList& examples = vector.getExamples(binIndex);

        for (auto it = examples.cbegin(); it != examples.cend(); it++) {
            BinVector::Example example = *it;
            histogramBuilderPtr->addToBin(binIndex, example.index);
        }
    }

    cacheEntry.histogramPtr = std::move(histogramBuilderPtr->build());
}

/**
 * Provides access to the thresholds that result from applying a binning method to the feature values of the training
 * examples.
 */
class ApproximateThresholds final : public AbstractThresholds {

    private:

        /**
         * Provides access to a subset of the thresholds that are stored by an instance of the class
         * `ApproximateThresholds`.
         */
        class ThresholdsSubset final : public IThresholdsSubset {

            private:

                /**
                 * A callback that allows to retrieve bins and corresponding statistics. If available, the bins and
                 * statistics are retrieved from the cache. Otherwise, they are computed by fetching the feature values
                 * from the feature matrix and applying a binning method.
                 */
                class Callback final : public IBinningObserver<float32>, public IRuleRefinementCallback<BinVector> {

                    private:

                        ThresholdsSubset& thresholdsSubset_;

                        uint32 featureIndex_;

                        BinVector* currentBinVector_;

                    public:

                        /**
                         * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the
                         *                          bins
                         * @param featureIndex      The index of the feature for which the bins should be retrieved
                         */
                        Callback(ThresholdsSubset& thresholdsSubset, uint32 featureIndex)
                            : thresholdsSubset_(thresholdsSubset), featureIndex_(featureIndex) {

                        }

                        std::unique_ptr<Result> get() override {
                            auto cacheFilteredIterator = thresholdsSubset_.cacheFiltered_.find(featureIndex_);
                            FilteredBinCacheEntry& cacheEntry = cacheFilteredIterator->second;
                            BinVector* binVector = cacheEntry.vectorPtr.get();
                            IHistogram* histogram = cacheEntry.histogramPtr.get();

                            if (binVector == nullptr) {
                                auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                                binVector = cacheIterator->second.get();

                                if (binVector == nullptr) {
                                    // Fetch feature vector...
                                    std::unique_ptr<FeatureVector> featureVectorPtr;
                                    thresholdsSubset_.thresholds_.featureMatrixPtr_->fetchFeatureVector(
                                        featureIndex_, featureVectorPtr);

                                    // Apply binning method...
                                    IFeatureBinning::FeatureInfo featureInfo =
                                        thresholdsSubset_.thresholds_.binningPtr_->getFeatureInfo(*featureVectorPtr);
                                    uint32 numBins = featureInfo.numBins;
                                    cacheIterator->second = std::move(std::make_unique<BinVector>(numBins, true));
                                    binVector = cacheIterator->second.get();
                                    currentBinVector_ = binVector;
                                    thresholdsSubset_.thresholds_.binningPtr_->createBins(featureInfo,
                                                                                          *featureVectorPtr, *this);
                                }

                                // Build histogram...
                                buildHistogram(*binVector, *thresholdsSubset_.thresholds_.statisticsPtr_, cacheEntry);
                                histogram = cacheEntry.histogramPtr.get();
                            }

                            // Filter bins, if necessary...
                            uint32 numConditions = thresholdsSubset_.numModifications_;

                            if (numConditions > cacheEntry.numConditions) {
                                filterAnyVector(*binVector, cacheEntry, numConditions, thresholdsSubset_.coverageMask_);
                                binVector = cacheEntry.vectorPtr.get();
                            }

                            return std::make_unique<Result>(*histogram, *binVector);
                        }

                        void onBinUpdate(uint32 binIndex, uint32 originalIndex, float32 value) override {
                            BinVector::bin_iterator binIterator = currentBinVector_->bins_begin();
                            binIterator[binIndex].numExamples += 1;

                            if (value < binIterator[binIndex].minValue) {
                                binIterator[binIndex].minValue = value;
                            }

                            if (binIterator[binIndex].maxValue < value) {
                                binIterator[binIndex].maxValue = value;
                            }

                            IndexedValue<float32> example;
                            example.index = originalIndex;
                            example.value = value;
                            BinVector::ExampleList& examples = currentBinVector_->getExamples(binIndex);
                            examples.push_front(example);
                        }

                };

                ApproximateThresholds& thresholds_;

                const IWeightVector& weights_;

                CoverageMask coverageMask_;

                uint32 numModifications_;

                std::unordered_map<uint32, FilteredBinCacheEntry> cacheFiltered_;

                template<class T>
                std::unique_ptr<IRuleRefinement> createApproximateRuleRefinement(const T& labelIndices,
                                                                                 uint32 featureIndex) {
                    auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredBinCacheEntry()).first;
                    BinVector* binVector = cacheFilteredIterator->second.vectorPtr.get();

                    if (binVector == nullptr) {
                        thresholds_.cache_.emplace(featureIndex, std::unique_ptr<BinVector>());
                    }

                    std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex);
                    std::unique_ptr<IHeadRefinement> headRefinementPtr =
                        thresholds_.headRefinementFactoryPtr_->create(labelIndices);
                    return std::make_unique<ApproximateRuleRefinement<T>>(std::move(headRefinementPtr), labelIndices,
                                                                          featureIndex, std::move(callbackPtr));
                }

            public:

                /**
                 * @param thresholds    A reference to an object of type `ApproximateThresholds` that stores the 
                 *                      thresholds
                 * @param weights       A reference to an object of type `IWeightWeight` that provides access to the
                 *                      weights of individual training examples
                 */
                ThresholdsSubset(ApproximateThresholds& thresholds, const IWeightVector& weights)
                    : thresholds_(thresholds), weights_(weights), 
                      coverageMask_(CoverageMask(thresholds.getNumExamples())), numModifications_(0) {

                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const FullIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createApproximateRuleRefinement(labelIndices, featureIndex);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createApproximateRuleRefinement(labelIndices, featureIndex);
                }

                void filterThresholds(Refinement& refinement) override {
                    numModifications_++;

                    uint32 featureIndex = refinement.featureIndex;
                    auto cacheFilteredIterator = cacheFiltered_.find(featureIndex);
                    FilteredBinCacheEntry& cacheEntry = cacheFilteredIterator->second;
                    BinVector* binVector = cacheEntry.vectorPtr.get();

                    if (binVector == nullptr) {
                        auto cacheIterator = thresholds_.cache_.find(featureIndex);
                        binVector = cacheIterator->second.get();
                    }

                    filterCurrentVector(*binVector, cacheEntry, refinement.end, refinement.covered, numModifications_,
                                        coverageMask_);
                }

                void filterThresholds(const Condition& condition) override {
                    // TODO Implement
                }

                void resetThresholds() override {
                    numModifications_ = 0;
                    cacheFiltered_.clear();
                    coverageMask_.reset();
                }

                const CoverageMask& getCoverageMask() const {
                    return coverageMask_;
                }

                float64 evaluateOutOfSample(const CoverageMask& coverageMask,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(*thresholds_.statisticsPtr_, 
                                                         *thresholds_.headRefinementFactoryPtr_, weights_, coverageMask,
                                                         head);
                }

                void recalculatePrediction(const CoverageMask& coverageMask, Refinement& refinement) const override {
                    recalculatePredictionInternally(*thresholds_.statisticsPtr_, *thresholds_.headRefinementFactoryPtr_,
                                                    coverageMask, refinement);
                }

                void applyPrediction(const AbstractPrediction& prediction) override {
                    updateStatisticsInternally(*thresholds_.statisticsPtr_, coverageMask_, prediction);
                }

        };

        std::shared_ptr<IFeatureBinning> binningPtr_;

        std::unordered_map<uint32, std::unique_ptr<BinVector>> cache_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsPtr             A shared pointer to an object of type `IStatistics` that provides access to
         *                                  statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         * @param binningPtr                A shared pointer to an object of type `IFeatureBinning` that implements the
         *                                  binning method to be used
         */
        ApproximateThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                              std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                              std::shared_ptr<IStatistics> statisticsPtr,
                              std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                              std::shared_ptr<IFeatureBinning> binningPtr)
            : AbstractThresholds(featureMatrixPtr, nominalFeatureMaskPtr, statisticsPtr, headRefinementFactoryPtr),
              binningPtr_(binningPtr) {

        }

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override {
            return std::make_unique<ApproximateThresholds::ThresholdsSubset>(*this, weights);
        }

};

ApproximateThresholdsFactory::ApproximateThresholdsFactory(std::shared_ptr<IFeatureBinning> binningPtr)
    : binningPtr_(binningPtr) {

}

std::unique_ptr<IThresholds> ApproximateThresholdsFactory::create(
        std::shared_ptr<IFeatureMatrix> featureMatrixPtr, std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
        std::shared_ptr<IStatistics> statisticsPtr,
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr) const {
    return std::make_unique<ApproximateThresholds>(featureMatrixPtr, nominalFeatureMaskPtr, statisticsPtr,
                                                   headRefinementFactoryPtr, binningPtr_);
}
