#include "mlrl/common/rule_refinement/feature_space_tabular.hpp"

#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/rule_refinement/rule_refinement_feature_based.hpp"
#include "mlrl/common/util/openmp.hpp"

#include <unordered_map>

template<typename IndexIterator, typename WeightVector>
static inline Quality evaluateOutOfSampleInternally(IndexIterator indexIterator, uint32 numExamples,
                                                    const WeightVector& weights, const CoverageMask& coverageMask,
                                                    const IStatistics& statistics, const IPrediction& prediction) {
    OutOfSampleWeightVector<WeightVector> outOfSampleWeights(weights);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr =
      prediction.createStatisticsSubset(statistics, outOfSampleWeights);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];

        if (statisticsSubsetPtr->hasNonZeroWeight(exampleIndex) && coverageMask[exampleIndex]) {
            statisticsSubsetPtr->addToSubset(exampleIndex);
        }
    }

    return statisticsSubsetPtr->calculateScores();
}

template<typename IndexIterator>
static inline void recalculatePredictionInternally(IndexIterator indexIterator, uint32 numExamples,
                                                   const CoverageMask& coverageMask, const IStatistics& statistics,
                                                   IPrediction& prediction) {
    EqualWeightVector weights(numExamples);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createStatisticsSubset(statistics, weights);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];

        if (coverageMask[exampleIndex]) {
            statisticsSubsetPtr->addToSubset(exampleIndex);
        }
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();
    scoreVector.updatePrediction(prediction);
}

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
 * Provides access to a tabular feature space.
 */
class TabularFeatureSpace final : public IFeatureSpace {
    private:

        /**
         * Provides access to a subset of a `TabularFeatureSpace`.
         *
         * @tparam WeightVector The type of the vector that provides access to the weights of individual training
         *                      examples
         */
        template<typename WeightVector>
        class FeatureSubspace final : public IFeatureSubspace {
            private:

                /**
                 * A callback that allows to retrieve feature vectors. If available, the feature vectors are retrieved
                 * from the cache. Otherwise, they are fetched from the feature matrix.
                 */
                class Callback final : public IFeatureSubspace::ICallback {
                    private:

                        FeatureSubspace& featureSubspace_;

                        const IFeatureInfo& featureInfo_;

                        const uint32 featureIndex_;

                    public:

                        /**
                         * @param featureSubspace   A reference to an object of type `FeatureSubspace` that caches the
                         *                          feature vectors
                         * @param featureInfo       A reference to an object of type `IFeatureInfo` that provides
                         *                          information about the types of individual features
                         * @param featureIndex      The index of the feature for which the feature vector should be
                         *                          retrieved
                         */
                        Callback(FeatureSubspace& featureSubspace, const IFeatureInfo& featureInfo, uint32 featureIndex)
                            : featureSubspace_(featureSubspace), featureInfo_(featureInfo),
                              featureIndex_(featureIndex) {}

                        Result get() override {
                            auto cacheFilteredIterator = featureSubspace_.cacheFiltered_.find(featureIndex_);
                            FilteredCacheEntry& cacheEntry = cacheFilteredIterator->second;
                            IFeatureVector* featureVector = cacheEntry.vectorPtr.get();

                            if (!featureVector) {
                                auto cacheIterator = featureSubspace_.featureSpace_.cache_.find(featureIndex_);
                                featureVector = cacheIterator->second.get();

                                if (!featureVector) {
                                    std::unique_ptr<IFeatureType> featureTypePtr = featureInfo_.createFeatureType(
                                      featureIndex_, featureSubspace_.featureSpace_.featureBinningFactory_);
                                    cacheIterator->second =
                                      featureSubspace_.featureSpace_.featureMatrix_.createFeatureVector(
                                        featureIndex_, *featureTypePtr);
                                    featureVector = cacheIterator->second.get();
                                }
                            }

                            // Filter feature vector, if only a subset of its elements are covered by the current
                            // rule...
                            uint32 numConditions = featureSubspace_.numModifications_;

                            if (numConditions > cacheEntry.numConditions) {
                                cacheEntry.vectorPtr = featureVector->createFilteredFeatureVector(
                                  cacheEntry.vectorPtr, featureSubspace_.coverageMask_);
                                cacheEntry.numConditions = numConditions;
                                featureVector = cacheEntry.vectorPtr.get();
                            }

                            return Result(*featureSubspace_.weightedStatisticsPtr_, *featureVector);
                        }
                };

                TabularFeatureSpace& featureSpace_;

                std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr_;

                const WeightVector& weights_;

                uint32 numCovered_;

                CoverageMask coverageMask_;

                uint32 numModifications_;

                std::unordered_map<uint32, FilteredCacheEntry> cacheFiltered_;

                template<typename IndexVector>
                std::unique_ptr<IRuleRefinement> createRuleRefinementInternally(const IndexVector& outputIndices,
                                                                                uint32 featureIndex) {
                    // Retrieve the `FilteredCacheEntry` from the cache, or insert a new one if it does not already
                    // exist...
                    auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry()).first;
                    IFeatureVector* featureVector = cacheFilteredIterator->second.vectorPtr.get();

                    // If the `FilteredCacheEntry` in the cache does not refer to an `IFeatureVector`, add an empty
                    // `unique_ptr` to the cache...
                    if (!featureVector) {
                        featureSpace_.cache_.emplace(featureIndex, std::unique_ptr<IFeatureVector>());
                    }

                    std::unique_ptr<Callback> callbackPtr =
                      std::make_unique<Callback>(*this, featureSpace_.featureInfo_, featureIndex);
                    return std::make_unique<FeatureBasedRuleRefinement<IndexVector>>(
                      outputIndices, featureIndex, numCovered_, std::move(callbackPtr));
                }

            public:

                /**
                 * @param featureSpace          A reference to an object of type `TabularFeatureSpace`, the subspace has
                 *                              been created from
                 * @param weightedStatisticsPtr An unique pointer to an object of type `IWeightedStatistics` that
                 *                              provides access to the statistics
                 * @param weights               A reference to an object of template type `WeightVector` that provides
                 *                              access to the weights of individual training examples
                 */
                FeatureSubspace(TabularFeatureSpace& featureSpace,
                                std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr, const WeightVector& weights)
                    : featureSpace_(featureSpace), weightedStatisticsPtr_(std::move(weightedStatisticsPtr)),
                      weights_(weights), numCovered_(weights.getNumNonZeroWeights()),
                      coverageMask_(featureSpace.featureMatrix_.getNumExamples()), numModifications_(0) {}

                /**
                 * @param other A reference to an object of type `FeatureSubspace` to be copied
                 */
                FeatureSubspace(const FeatureSubspace& other)
                    : featureSpace_(other.featureSpace_), weightedStatisticsPtr_(other.weightedStatisticsPtr_->copy()),
                      weights_(other.weights_), numCovered_(other.numCovered_), coverageMask_(other.coverageMask_),
                      numModifications_(other.numModifications_) {}

                std::unique_ptr<IFeatureSubspace> copy() const override {
                    return std::make_unique<FeatureSubspace<WeightVector>>(*this);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const CompleteIndexVector& outputIndices,
                                                                      uint32 featureIndex) override {
                    return createRuleRefinementInternally(outputIndices, featureIndex);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& outputIndices,
                                                                      uint32 featureIndex) override {
                    return createRuleRefinementInternally(outputIndices, featureIndex);
                }

                void filterSubspace(const Condition& condition) override {
                    uint32 featureIndex = condition.featureIndex;
                    auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry()).first;
                    FilteredCacheEntry& cacheEntry = cacheFilteredIterator->second;
                    IFeatureVector* featureVector = cacheEntry.vectorPtr.get();

                    if (!featureVector) {
                        auto cacheIterator =
                          featureSpace_.cache_.emplace(featureIndex, std::unique_ptr<IFeatureVector>()).first;
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
                    numCovered_ = condition.numCovered;
                    featureVector->updateCoverageMaskAndStatistics(condition, coverageMask_, numModifications_,
                                                                   *weightedStatisticsPtr_);
                    cacheEntry.vectorPtr = featureVector->createFilteredFeatureVector(cacheEntry.vectorPtr, condition);
                    cacheEntry.numConditions = numModifications_;
                }

                void resetSubspace() override {
                    numModifications_ = 0;
                    numCovered_ = weights_.getNumNonZeroWeights();
                    cacheFiltered_.clear();
                    coverageMask_.reset();
                }

                const CoverageMask& getCoverageMask() const override {
                    return coverageMask_;
                }

                Quality evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageMask,
                                            const IPrediction& head) const override {
                    return evaluateOutOfSampleInternally<SinglePartition::const_iterator>(
                      partition.cbegin(), partition.getNumElements(), weights_, coverageMask,
                      featureSpace_.statisticsProvider_.get(), head);
                }

                Quality evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageMask,
                                            const IPrediction& head) const override {
                    return evaluateOutOfSampleInternally<BiPartition::const_iterator>(
                      partition.first_cbegin(), partition.getNumFirst(), weights_, coverageMask,
                      featureSpace_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageMask,
                                           IPrediction& head) const override {
                    recalculatePredictionInternally<SinglePartition::const_iterator>(
                      partition.cbegin(), partition.getNumElements(), coverageMask,
                      featureSpace_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageMask,
                                           IPrediction& head) const override {
                    recalculatePredictionInternally<BiPartition::const_iterator>(
                      partition.first_cbegin(), partition.getNumFirst(), coverageMask,
                      featureSpace_.statisticsProvider_.get(), head);
                }

                void applyPrediction(const IPrediction& prediction) override {
                    IStatistics& statistics = featureSpace_.statisticsProvider_.get();
                    uint32 numStatistics = statistics.getNumStatistics();
                    const CoverageMask* coverageMaskPtr = &coverageMask_;
                    std::unique_ptr<IStatisticsUpdate> statisticsUpdatePtr =
                      prediction.createStatisticsUpdate(statistics);
                    IStatisticsUpdate* statisticsUpdateRawPtr = statisticsUpdatePtr.get();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numStatistics) firstprivate(coverageMaskPtr) \
      firstprivate(statisticsUpdateRawPtr) schedule(dynamic) \
      num_threads(featureSpace_.multiThreadingSettings_.numThreads)
#endif
                    for (int64 i = 0; i < numStatistics; i++) {
                        if ((*coverageMaskPtr)[i]) {
                            statisticsUpdateRawPtr->applyPrediction(i);
                        }
                    }
                }

                void revertPrediction(const IPrediction& prediction) override {
                    IStatistics& statistics = featureSpace_.statisticsProvider_.get();
                    uint32 numStatistics = statistics.getNumStatistics();
                    const CoverageMask* coverageMaskPtr = &coverageMask_;
                    std::unique_ptr<IStatisticsUpdate> statisticsUpdatePtr =
                      prediction.createStatisticsUpdate(statistics);
                    IStatisticsUpdate* statisticsUpdateRawPtr = statisticsUpdatePtr.get();

#if MULTI_THREADING_SUPPORT_ENABLED
    #pragma omp parallel for firstprivate(numStatistics) firstprivate(coverageMaskPtr) \
      firstprivate(statisticsUpdateRawPtr) schedule(dynamic) \
      num_threads(featureSpace_.multiThreadingSettings_.numThreads)
#endif
                    for (int64 i = 0; i < numStatistics; i++) {
                        if ((*coverageMaskPtr)[i]) {
                            statisticsUpdateRawPtr->revertPrediction(i);
                        }
                    }
                }
        };

        const IColumnWiseFeatureMatrix& featureMatrix_;

        const IFeatureInfo& featureInfo_;

        IStatisticsProvider& statisticsProvider_;

        const IFeatureBinningFactory& featureBinningFactory_;

        const MultiThreadingSettings multiThreadingSettings_;

        std::unordered_map<uint32, std::unique_ptr<IFeatureVector>> cache_;

    public:

        /**
         * @param featureMatrix             A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                                  column-wise access to the feature values of individual training examples
         * @param featureInfo               A reference to an object of type `IFeatureInfo` that provides information
         *                                  about the types of individual features
         * @param statisticsProvider        A reference to an object of type `IStatisticsProvider` that provides access
         *                                  to statistics about the quality of predictions for training examples
         * @param featureBinningFactory     A reference to an object of type `IFeatureBinningFactory` that allows to
         *                                  create implementations of the binning method to be used for assigning
         *                                  numerical feature values to bins
         * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to be
         *                                  used for updating statistics in parallel
         */
        TabularFeatureSpace(const IColumnWiseFeatureMatrix& featureMatrix, const IFeatureInfo& featureInfo,
                            IStatisticsProvider& statisticsProvider,
                            const IFeatureBinningFactory& featureBinningFactory,
                            MultiThreadingSettings multiThreadingSettings)
            : featureMatrix_(featureMatrix), featureInfo_(featureInfo), statisticsProvider_(statisticsProvider),
              featureBinningFactory_(featureBinningFactory), multiThreadingSettings_(multiThreadingSettings) {}

        IStatisticsProvider& getStatisticsProvider() const override final {
            return statisticsProvider_;
        }

        std::unique_ptr<IFeatureSubspace> createSubspace(const EqualWeightVector& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<TabularFeatureSpace::FeatureSubspace<EqualWeightVector>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }

        std::unique_ptr<IFeatureSubspace> createSubspace(const BitWeightVector& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<TabularFeatureSpace::FeatureSubspace<BitWeightVector>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }

        std::unique_ptr<IFeatureSubspace> createSubspace(const DenseWeightVector<uint32>& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<TabularFeatureSpace::FeatureSubspace<DenseWeightVector<uint32>>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }
};

TabularFeatureSpaceFactory::TabularFeatureSpaceFactory(std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr,
                                                       MultiThreadingSettings multiThreadingSettings)
    : featureBinningFactoryPtr_(std::move(featureBinningFactoryPtr)), multiThreadingSettings_(multiThreadingSettings) {}

std::unique_ptr<IFeatureSpace> TabularFeatureSpaceFactory::create(const IColumnWiseFeatureMatrix& featureMatrix,
                                                                  const IFeatureInfo& featureInfo,
                                                                  IStatisticsProvider& statisticsProvider) const {
    return std::make_unique<TabularFeatureSpace>(featureMatrix, featureInfo, statisticsProvider,
                                                 *featureBinningFactoryPtr_, multiThreadingSettings_);
}
