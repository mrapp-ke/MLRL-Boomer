#include "mlrl/common/thresholds/thresholds_exact.hpp"

#include "mlrl/common/rule_refinement/rule_refinement_exact.hpp"
#include "mlrl/common/util/openmp.hpp"
#include "thresholds_common.hpp"

#include <unordered_map>

/**
 * An entry that is stored in a cache and contains an unique pointer to a feature vector. The field `numConditions`
 * specifies how many conditions the rule contained when the vector was updated for the last time. It may be used to
 * check if the vector is still valid or must be updated.
 */
struct FilteredCacheEntry final {
    public:

        FilteredCacheEntry() : numConditions(0) {}

        /**
         * An unique pointer to an object of type `IFeatureVector` that stores feature values.
         */
        std::unique_ptr<IFeatureVector> vectorPtr;

        /**
         * The number of conditions that were contained by the rule when the cache was updated for the last time.
         */
        uint32 numConditions;
};

/**
 * Provides access to all thresholds that result from the feature values of the training examples.
 */
class ExactThresholds final : public AbstractThresholds {
    private:

        /**
         * Provides access to a subset of the thresholds that are stored by an instance of the class `ExactThresholds`.
         *
         * @tparam WeightVector The type of the vector that provides access to the weights of individual training
         *                      examples
         */
        template<typename WeightVector>
        class ThresholdsSubset final : public IThresholdsSubset {
            private:

                /**
                 * A callback that allows to retrieve feature vectors. If available, the feature vectors are retrieved
                 * from the cache. Otherwise, they are fetched from the feature matrix.
                 */
                class Callback final : public IRuleRefinementCallback<IImmutableWeightedStatistics, IFeatureVector> {
                    private:

                        ThresholdsSubset& thresholdsSubset_;

                        const IFeatureInfo& featureInfo_;

                        const uint32 featureIndex_;

                    public:

                        /**
                         * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the
                         *                          feature vectors
                         * @param featureInfo       A reference to an object of type `IFeatureInfo` that provides
                         *                          information about the types of individual features
                         * @param featureIndex      The index of the feature for which the feature vector should be
                         *                          retrieved
                         */
                        Callback(ThresholdsSubset& thresholdsSubset, const IFeatureInfo& featureInfo,
                                 uint32 featureIndex)
                            : thresholdsSubset_(thresholdsSubset), featureInfo_(featureInfo),
                              featureIndex_(featureIndex) {}

                        Result get() override {
                            auto cacheFilteredIterator = thresholdsSubset_.cacheFiltered_.find(featureIndex_);
                            FilteredCacheEntry& cacheEntry = cacheFilteredIterator->second;
                            IFeatureVector* featureVector = cacheEntry.vectorPtr.get();

                            if (!featureVector) {
                                auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                                featureVector = cacheIterator->second.get();

                                if (!featureVector) {
                                    std::unique_ptr<IFeatureType> featureTypePtr =
                                      featureInfo_.createFeatureType(featureIndex_);
                                    cacheIterator->second =
                                      thresholdsSubset_.thresholds_.featureMatrix_.createFeatureVector(featureIndex_,
                                                                                                       *featureTypePtr);
                                    featureVector = cacheIterator->second.get();
                                }
                            }

                            // Filter feature vector, if only a subset of its elements are covered by the current
                            // rule...
                            uint32 numConditions = thresholdsSubset_.numModifications_;

                            if (numConditions > cacheEntry.numConditions) {
                                cacheEntry.vectorPtr = featureVector->createFilteredFeatureVector(
                                  cacheEntry.vectorPtr, thresholdsSubset_.coverageMask_);
                                cacheEntry.numConditions = numConditions;
                                featureVector = cacheEntry.vectorPtr.get();
                            }

                            return Result(*thresholdsSubset_.weightedStatisticsPtr_, *featureVector);
                        }
                };

                ExactThresholds& thresholds_;

                std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr_;

                const WeightVector& weights_;

                CoverageMask coverageMask_;

                uint32 numModifications_;

                std::unordered_map<uint32, FilteredCacheEntry> cacheFiltered_;

                template<typename IndexVector>
                std::unique_ptr<IRuleRefinement> createExactRuleRefinement(const IndexVector& labelIndices,
                                                                           uint32 featureIndex) {
                    // Retrieve the `FilteredCacheEntry` from the cache, or insert a new one if it does not already
                    // exist...
                    auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry()).first;
                    IFeatureVector* featureVector = cacheFilteredIterator->second.vectorPtr.get();

                    // If the `FilteredCacheEntry` in the cache does not refer to an `IFeatureVector`, add an empty
                    // `unique_ptr` to the cache...
                    if (!featureVector) {
                        thresholds_.cache_.emplace(featureIndex, std::unique_ptr<IFeatureVector>());
                    }

                    std::unique_ptr<Callback> callbackPtr =
                      std::make_unique<Callback>(*this, thresholds_.featureInfo_, featureIndex);
                    return std::make_unique<ExactRuleRefinement<IndexVector>>(labelIndices, featureIndex,
                                                                              std::move(callbackPtr));
                }

            public:

                /**
                 * @param thresholds            A reference to an object of type `ExactThresholds` that stores the
                 *                              thresholds
                 * @param weightedStatisticsPtr An unique pointer to an object of type `IWeightedStatistics` that
                 *                              provides access to the statistics
                 * @param weights               A reference to an object of template type `WeightVector` that provides
                 *                              access to the weights of individual training examples
                 */
                ThresholdsSubset(ExactThresholds& thresholds,
                                 std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr,
                                 const WeightVector& weights)
                    : thresholds_(thresholds), weightedStatisticsPtr_(std::move(weightedStatisticsPtr)),
                      weights_(weights), coverageMask_(thresholds.featureMatrix_.getNumExamples()),
                      numModifications_(0) {}

                /**
                 * @param thresholdsSubset A reference to an object of type `ThresholdsSubset` to be copied
                 */
                ThresholdsSubset(const ThresholdsSubset& thresholdsSubset)
                    : thresholds_(thresholdsSubset.thresholds_),
                      weightedStatisticsPtr_(thresholdsSubset.weightedStatisticsPtr_->copy()),
                      weights_(thresholdsSubset.weights_), coverageMask_(thresholdsSubset.coverageMask_),
                      numModifications_(thresholdsSubset.numModifications_) {}

                std::unique_ptr<IThresholdsSubset> copy() const override {
                    return std::make_unique<ThresholdsSubset<WeightVector>>(*this);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const CompleteIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createExactRuleRefinement(labelIndices, featureIndex);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createExactRuleRefinement(labelIndices, featureIndex);
                }

                void filterThresholds(const Condition& condition) override {
                    uint32 featureIndex = condition.featureIndex;
                    auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry()).first;
                    FilteredCacheEntry& cacheEntry = cacheFilteredIterator->second;
                    IFeatureVector* featureVector = cacheEntry.vectorPtr.get();

                    if (!featureVector) {
                        auto cacheIterator =
                          thresholds_.cache_.emplace(featureIndex, std::unique_ptr<IFeatureVector>()).first;
                        featureVector = cacheIterator->second.get();
                    }

                    // Identify the examples that are covered by the condition...
                    if (numModifications_ > cacheEntry.numConditions) {
                        cacheEntry.vectorPtr =
                          featureVector->createFilteredFeatureVector(cacheEntry.vectorPtr, coverageMask_);
                        cacheEntry.numConditions = numModifications_;
                        featureVector = cacheEntry.vectorPtr.get();
                    }

                    numModifications_++;
                    featureVector->updateCoverageMaskAndStatistics(condition, coverageMask_, numModifications_,
                                                                   *weightedStatisticsPtr_);
                    cacheEntry.vectorPtr = featureVector->createFilteredFeatureVector(cacheEntry.vectorPtr, condition);
                    cacheEntry.numConditions = numModifications_;
                }

                void resetThresholds() override {
                    numModifications_ = 0;
                    cacheFiltered_.clear();
                    coverageMask_.reset();
                }

                const ICoverageState& getCoverageState() const override {
                    return coverageMask_;
                }

                Quality evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageState,
                                            const IPrediction& head) const override {
                    return evaluateOutOfSampleInternally<SinglePartition::const_iterator>(
                      partition.cbegin(), partition.getNumElements(), weights_, coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                Quality evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageState,
                                            const IPrediction& head) const override {
                    return evaluateOutOfSampleInternally<BiPartition::const_iterator>(
                      partition.first_cbegin(), partition.getNumFirst(), weights_, coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                Quality evaluateOutOfSample(const SinglePartition& partition, const CoverageSet& coverageState,
                                            const IPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, thresholds_.statisticsProvider_.get(),
                                                         head);
                }

                Quality evaluateOutOfSample(BiPartition& partition, const CoverageSet& coverageState,
                                            const IPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, partition,
                                                         thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageState,
                                           IPrediction& head) const override {
                    recalculatePredictionInternally<SinglePartition::const_iterator>(
                      partition.cbegin(), partition.getNumElements(), coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageState,
                                           IPrediction& head) const override {
                    recalculatePredictionInternally<BiPartition::const_iterator>(
                      partition.first_cbegin(), partition.getNumFirst(), coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageSet& coverageState,
                                           IPrediction& head) const override {
                    recalculatePredictionInternally(coverageState, thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(BiPartition& partition, const CoverageSet& coverageState,
                                           IPrediction& head) const override {
                    recalculatePredictionInternally(coverageState, partition, thresholds_.statisticsProvider_.get(),
                                                    head);
                }

                void applyPrediction(const IPrediction& prediction) override {
                    IStatistics& statistics = thresholds_.statisticsProvider_.get();
                    uint32 numStatistics = statistics.getNumStatistics();
                    const CoverageMask* coverageMaskPtr = &coverageMask_;
                    const IPrediction* predictionPtr = &prediction;
                    IStatistics* statisticsPtr = &statistics;

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numStatistics) firstprivate(coverageMaskPtr) firstprivate(predictionPtr) \
      firstprivate(statisticsPtr) schedule(dynamic) num_threads(thresholds_.numThreads_)
#endif
                    for (int64 i = 0; i < numStatistics; i++) {
                        if (coverageMaskPtr->isCovered(i)) {
                            predictionPtr->apply(*statisticsPtr, i);
                        }
                    }
                }

                void revertPrediction(const IPrediction& prediction) override {
                    IStatistics& statistics = thresholds_.statisticsProvider_.get();
                    uint32 numStatistics = statistics.getNumStatistics();
                    const CoverageMask* coverageMaskPtr = &coverageMask_;
                    const IPrediction* predictionPtr = &prediction;
                    IStatistics* statisticsPtr = &statistics;

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numStatistics) firstprivate(coverageMaskPtr) firstprivate(predictionPtr) \
      firstprivate(statisticsPtr) schedule(dynamic) num_threads(thresholds_.numThreads_)
#endif
                    for (int64 i = 0; i < numStatistics; i++) {
                        if (coverageMaskPtr->isCovered(i)) {
                            predictionPtr->revert(*statisticsPtr, i);
                        }
                    }
                }
        };

        const uint32 numThreads_;

        std::unordered_map<uint32, std::unique_ptr<IFeatureVector>> cache_;

    public:

        /**
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of individual training examples
         * @param featureInfo           A reference to an object of type `IFeatureInfo` that provides information about
         *                              the types of individual features
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the labels of the training examples
         * @param numThreads            The number of CPU threads to be used to update statistics in parallel
         */
        ExactThresholds(const IColumnWiseFeatureMatrix& featureMatrix, const IFeatureInfo& featureInfo,
                        IStatisticsProvider& statisticsProvider, uint32 numThreads)
            : AbstractThresholds(featureMatrix, featureInfo, statisticsProvider), numThreads_(numThreads) {}

        std::unique_ptr<IThresholdsSubset> createSubset(const EqualWeightVector& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<ExactThresholds::ThresholdsSubset<EqualWeightVector>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }

        std::unique_ptr<IThresholdsSubset> createSubset(const BitWeightVector& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<ExactThresholds::ThresholdsSubset<BitWeightVector>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }

        std::unique_ptr<IThresholdsSubset> createSubset(const DenseWeightVector<uint32>& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<ExactThresholds::ThresholdsSubset<DenseWeightVector<uint32>>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }
};

ExactThresholdsFactory::ExactThresholdsFactory(uint32 numThreads) : numThreads_(numThreads) {}

std::unique_ptr<IThresholds> ExactThresholdsFactory::create(const IColumnWiseFeatureMatrix& featureMatrix,
                                                            const IFeatureInfo& featureInfo,
                                                            IStatisticsProvider& statisticsProvider) const {
    return std::make_unique<ExactThresholds>(featureMatrix, featureInfo, statisticsProvider, numThreads_);
}
