#include "common/thresholds/thresholds_approximate.hpp"
#include "common/binning/bin_vector.hpp"
#include "common/binning/feature_binning_nominal.hpp"
#include "common/rule_refinement/rule_refinement_approximate.hpp"
#include "common/data/arrays.hpp"
#include "thresholds_common.hpp"
#include <unordered_map>
#include <cmath>


/**
 * A vector that stores the indices of the bins, individual examples belong to.
 */
typedef DenseVector<uint32> BinIndexVector;

/**
 * An entry that is stored in the cache and contains unique pointers to vectors that store bins and their corresponding
 * weights. Moreover, it contains unique pointers to a vector that stores the index of the bins individual examples
 * belong to, as well as to an histogram.
 */
struct CacheEntry {
    std::unique_ptr<BinVector> binVectorPtr;
    std::unique_ptr<BinIndexVector> binIndicesPtr;
    std::unique_ptr<IHistogram> histogramPtr;
    std::unique_ptr<BinWeightVector> weightVectorPtr;
};

static inline void addValueToBinVector(BinVector& binVector, uint32 binIndex, uint32 originalIndex, float64 value) {
    BinVector::iterator binIterator = binVector.begin();

    if (value < binIterator[binIndex].minValue) {
        binIterator[binIndex].minValue = value;
    }

    if (value > binIterator[binIndex].maxValue) {
        binIterator[binIndex].maxValue = value;
    }
}

/**
 * Removes all empty bins from a given `BinVector` and adjusts the indices of the bins, individual examples belong to,
 * accordingly.
 *
 * @param binVector     A reference to an object of type `BinVector`, the empty bins should be removed from
 * @param binIndices    A reference to an object of type `BinIndexVector` that stores the indices of the bins,
 *                      individual examples belong to
 */
static inline void removeEmptyBins(BinVector& binVector, BinIndexVector& binIndices) {
    uint32 numBins = binVector.getNumElements();
    BinVector::iterator binIterator = binVector.begin();
    uint32 mapping[numElements];
    uint32 n = 0;

    // Remove empty bins...
    for (uint32 i = 0; i < numBins; i++) {
        mapping[i] = n;
        float32 minValue = binIterator[i].minValue;

        if (std::isfinite(minValue)) {
            binIterator[n].minValue = minValue;
            binIterator[n].maxValue = binIterator[i].maxValue;
            n++;
        }
    }

    binVector.setNumElements(n, true);

    // Adjust bin indices...
    BinIndexVector::iterator indexIterator = binIndices.begin();
    uint32 numExamples = binIndices.getNumElements();

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 binIndex = indexIterator[i];
        indexIterator[i] = mapping[binIndex];
    }
}

static inline void updateCoveredExamples(const BinVector& binVector, const BinIndexVector& binIndices,
                                         intp conditionEnd, bool covered, CoverageSet& coverageSet,
                                         IStatistics& statistics, const IWeightVector& weights) {
    intp minBinIndex, maxBinIndex;

    if (covered) {
        minBinIndex = 0;
        maxBinIndex = conditionEnd - 1;
    } else {
        minBinIndex = conditionEnd;
        maxBinIndex = binVector.getNumElements() - 1;
    }

    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::iterator coverageSetIterator = coverageSet.begin();
    BinIndexVector::const_iterator indexIterator = binIndices.cbegin();
    statistics.resetCoveredStatistics();
    uint32 n = 0;

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];
        uint32 binIndex = indexIterator[exampleIndex];

        // Check if the example is still covered, i.e., if the corresponding bin is contained in the filtered bin
        // vector...
        if (binIndex >= minBinIndex && binIndex <= maxBinIndex) {
            uint32 weight = weights.getWeight(exampleIndex);
            statistics.updateCoveredStatistic(exampleIndex, weight, false);
            coverageSetIterator[n] = exampleIndex;
            n++;
        }
    }

    coverageSet.setNumCovered(n);
}

static inline void rebuildHistogram(const BinIndexVector& binIndices, BinWeightVector& binWeights,
                                    IHistogram& histogram, const IWeightVector& weights,
                                    const CoverageSet& coverageSet) {
    // Reset all statistics in the histogram to zero...
    histogram.setAllToZero();

    // Reset the weights of all bins to zero...
    BinWeightVector::iterator binWeightIterator = binWeights.begin();
    setArrayToZeros(binWeightIterator, binWeights.getNumElements());

    // Iterate the covered examples and add their statistics to the corresponding bin...
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator coverageSetIterator = coverageSet.cbegin();
    BinIndexVector::const_iterator binIndexIterator = binIndices.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];
        uint32 binIndex = binIndexIterator[exampleIndex];
        uint32 weight = weights.getWeight(exampleIndex);
        binWeightIterator[binIndex] += weight;
        histogram.addToBin(binIndex, exampleIndex, weight);
    }
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
                class Callback final : public IRuleRefinementCallback<BinVector, BinWeightVector> {

                    private:

                        ThresholdsSubset& thresholdsSubset_;

                        uint32 featureIndex_;

                        bool nominal_;

                    public:

                        /**
                         * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the
                         *                          bins
                         * @param featureIndex      The index of the feature for which the bins should be retrieved
                         * @param nominal           True, if the feature at index `featureIndex` is nominal, false
                         *                          otherwise
                         */
                        Callback(ThresholdsSubset& thresholdsSubset, uint32 featureIndex, bool nominal)
                            : thresholdsSubset_(thresholdsSubset), featureIndex_(featureIndex), nominal_(nominal) {

                        }

                        std::unique_ptr<Result> get() override {
                            auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                            BinVector* binVector = cacheIterator->second.binVectorPtr.get();
                            BinIndexVector* binIndices = cacheIterator->second.binIndicesPtr.get();

                            if (binVector == nullptr) {
                                // Fetch feature vector...
                                std::unique_ptr<FeatureVector> featureVectorPtr;
                                thresholdsSubset_.thresholds_.featureMatrixPtr_->fetchFeatureVector(featureIndex_,
                                                                                                    featureVectorPtr);
                                uint32 numExamples = featureVectorPtr->getNumElements();

                                // Apply binning method...
                                const IFeatureBinning& binning =
                                    nominal_ ? thresholdsSubset_.thresholds_.nominalBinning_
                                             : *thresholdsSubset_.thresholds_.binningPtr_;
                                IFeatureBinning::FeatureInfo featureInfo = binning.getFeatureInfo(*featureVectorPtr);
                                uint32 numBins = featureInfo.numBins;
                                cacheIterator->second.binVectorPtr = std::make_unique<BinVector>(numBins);
                                cacheIterator->second.binIndicesPtr = std::make_unique<BinIndexVector>(numExamples);
                                binVector = cacheIterator->second.binVectorPtr.get();
                                binIndices = cacheIterator->second.binIndicesPtr.get();
                                auto callback = [=](uint32 binIndex, uint32 originalIndex, float32 value) {
                                    binIndices->begin()[originalIndex] = binIndex;
                                    addValueToBinVector(*binVector, binIndex, originalIndex, value);
                                };
                                binning.createBins(featureInfo, *featureVectorPtr, callback);

                                if (!nominal_) {
                                    removeEmptyBins(*binVector, *binIndices);
                                    numBins = binVector->getNumElements();
                                }

                                cacheIterator->second.histogramPtr =
                                    thresholdsSubset_.thresholds_.statisticsProviderPtr_->get().createHistogram(
                                        numBins);
                                cacheIterator->second.weightVectorPtr = std::make_unique<BinWeightVector>(numBins);
                            }

                            // Rebuild histogram...
                            IHistogram& histogram = *cacheIterator->second.histogramPtr;
                            BinWeightVector& binWeights = *cacheIterator->second.weightVectorPtr;
                            rebuildHistogram(*binIndices, binWeights, histogram, thresholdsSubset_.weights_,
                                             thresholdsSubset_.coverageSet_);

                            return std::make_unique<Result>(histogram, binWeights, *binVector);
                        }

                };

                ApproximateThresholds& thresholds_;

                const IWeightVector& weights_;

                CoverageSet coverageSet_;

                template<class T>
                std::unique_ptr<IRuleRefinement> createApproximateRuleRefinement(const T& labelIndices,
                                                                                 uint32 featureIndex) {
                    thresholds_.cache_.emplace(featureIndex, CacheEntry());
                    bool nominal = thresholds_.nominalFeatureMaskPtr_->isNominal(featureIndex);
                    std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex, nominal);
                    std::unique_ptr<IHeadRefinement> headRefinementPtr =
                        thresholds_.headRefinementFactoryPtr_->create(labelIndices);
                    return std::make_unique<ApproximateRuleRefinement<T>>(std::move(headRefinementPtr), labelIndices,
                                                                          featureIndex, nominal,
                                                                          std::move(callbackPtr));
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
                      coverageSet_(CoverageSet(thresholds.getNumExamples())) {

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
                    uint32 featureIndex = refinement.featureIndex;
                    auto cacheIterator = thresholds_.cache_.find(featureIndex);
                    const BinVector& binVector = *cacheIterator->second.binVectorPtr;
                    const BinIndexVector& binIndices = *cacheIterator->second.binIndicesPtr;
                    updateCoveredExamples(binVector, binIndices, refinement.end, refinement.covered, coverageSet_,
                                          thresholds_.statisticsProviderPtr_->get(), weights_);
                }

                void filterThresholds(const Condition& condition) override {
                    // TODO Implement
                }

                void resetThresholds() override {
                    coverageSet_.reset();
                }

                const ICoverageState& getCoverageState() const {
                    return coverageSet_;
                }

                float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<SinglePartition::const_iterator>(
                        partition.cbegin(), partition.getNumElements(), weights_, coverageState,
                        thresholds_.statisticsProviderPtr_->get(), *thresholds_.headRefinementFactoryPtr_, head);
                }

                float64 evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<BiPartition::const_iterator>(
                        partition.first_cbegin(), partition.getNumFirst(), weights_, coverageState,
                        thresholds_.statisticsProviderPtr_->get(), *thresholds_.headRefinementFactoryPtr_, head);
                }

                float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState,
                                                         thresholds_.statisticsProviderPtr_->get(),
                                                         *thresholds_.headRefinementFactoryPtr_, head);
                }

                float64 evaluateOutOfSample(BiPartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, partition,
                                                         thresholds_.statisticsProviderPtr_->get(),
                                                         *thresholds_.headRefinementFactoryPtr_, head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally<SinglePartition::const_iterator>(
                        partition.cbegin(), partition.getNumElements(), coverageState,
                        thresholds_.statisticsProviderPtr_->get(), *thresholds_.headRefinementFactoryPtr_, refinement);
                }

                void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally<BiPartition::const_iterator>(
                        partition.first_cbegin(), partition.getNumFirst(), coverageState,
                        thresholds_.statisticsProviderPtr_->get(), *thresholds_.headRefinementFactoryPtr_, refinement);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally(coverageState, thresholds_.statisticsProviderPtr_->get(),
                                                    *thresholds_.headRefinementFactoryPtr_, refinement);
                }

                void recalculatePrediction(BiPartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally(coverageState, partition, thresholds_.statisticsProviderPtr_->get(),
                                                    *thresholds_.headRefinementFactoryPtr_, refinement);
                }

                void applyPrediction(const AbstractPrediction& prediction) override {
                    uint32 numCovered = coverageSet_.getNumCovered();
                    CoverageSet::const_iterator iterator = coverageSet_.cbegin();
                    const AbstractPrediction* predictionPtr = &prediction;
                    IStatistics* statisticsPtr = &thresholds_.statisticsProviderPtr_->get();
                    uint32 numThreads = thresholds_.numThreads_;

                    #pragma omp parallel for firstprivate(numCovered) firstprivate(iterator) \
                    firstprivate(predictionPtr) firstprivate(statisticsPtr) schedule(dynamic) num_threads(numThreads)
                    for (uint32 i = 0; i < numCovered; i++) {
                        uint32 exampleIndex = iterator[i];
                        predictionPtr->apply(*statisticsPtr, exampleIndex);
                    }
                }

        };

        NominalFeatureBinning nominalBinning_;

        std::shared_ptr<IFeatureBinning> binningPtr_;

        uint32 numThreads_;

        std::unordered_map<uint32, CacheEntry> cache_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsProviderPtr     A shared pointer to an object of type `IStatisticsProvider` that provides
         *                                  access to statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         * @param binningPtr                A shared pointer to an object of type `IFeatureBinning` that implements the
         *                                  binning method to be used
         * @param numThreads                The number of CPU threads to be used to update statistics in parallel
         */
        ApproximateThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                              std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                              std::shared_ptr<IStatisticsProvider> statisticsProviderPtr,
                              std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                              std::shared_ptr<IFeatureBinning> binningPtr, uint32 numThreads)
            : AbstractThresholds(featureMatrixPtr, nominalFeatureMaskPtr, statisticsProviderPtr,
                                 headRefinementFactoryPtr), binningPtr_(binningPtr), numThreads_(numThreads) {

        }

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override {
            updateSampledStatisticsInternally(statisticsProviderPtr_->get(), weights);
            return std::make_unique<ApproximateThresholds::ThresholdsSubset>(*this, weights);
        }

};

ApproximateThresholdsFactory::ApproximateThresholdsFactory(std::shared_ptr<IFeatureBinning> binningPtr,
                                                           uint32 numThreads)
    : binningPtr_(binningPtr), numThreads_(numThreads) {

}

std::unique_ptr<IThresholds> ApproximateThresholdsFactory::create(
        std::shared_ptr<IFeatureMatrix> featureMatrixPtr, std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
        std::shared_ptr<IStatisticsProvider> statisticsProviderPtr,
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr) const {
    return std::make_unique<ApproximateThresholds>(featureMatrixPtr, nominalFeatureMaskPtr, statisticsProviderPtr,
                                                   headRefinementFactoryPtr, binningPtr_, numThreads_);
}
