#include "thresholds_common.h"
#include "thresholds_approximate.h"
#include "rule_refinement_approximate.h"


static inline void filterCurrentVector(const BinVector& vector, FilteredCacheEntry<BinVector>& cacheEntry,
                                       intp conditionStart, intp conditionEnd, bool covered, uint32 numConditions,
                                       CoverageMask& coverageMask)
{
    //TODO: in this PR

    BinVector* filteredVector = cacheEntry.vectorPtr.get();

    typename BinVector::const_iterator iterator = vector.cbegin();
    BinVector::iterator filteredIterator = filteredVector->begin();
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    uint32 numTotalElements = vector.getNumElements();
    coverageMask.target = numConditions;
    intp start, end;
    start = conditionStart;
    end = conditionEnd;
    uint32 i = 0;

    if (covered) {
        for(intp r = 0; r < start; r++){
            coverageMaskIterator[r] = numConditions;
            filteredIterator[i].numExamples = iterator[r].numExamples;
            filteredIterator[i].minValue = iterator[r].minValue;
            filteredIterator[i].maxValue = iterator[r].maxValue;
            i++;
        }
        for(intp r = end; r <= numTotalElements; r++){
            coverageMaskIterator[r] = numConditions;
            filteredIterator[i].numExamples = iterator[r].numExamples;
            filteredIterator[i].minValue = iterator[r].minValue;
            filteredIterator[i].maxValue = iterator[r].maxValue;
            i++;
        }
    } else {
        for(intp r = start; r < end; r++) {
            coverageMaskIterator[r] = numConditions;
            filteredIterator[i].numExamples = iterator[r].numExamples;
            filteredIterator[i].minValue = iterator[r].minValue;
            filteredIterator[i].maxValue = iterator[r].maxValue;
            i++;
        }
    }
}

/**
 * Provides access to a subset of the thresholds that are stored by an instance of the class
 * `ApproximateThresholds`.
 */
class ApproximateThresholds::ThresholdsSubset : public IThresholdsSubset {

    private:

        /**
         * A callback that allows to retrieve bins and corresponding statistics. If available, the bins and statistics
         * are retrieved from the cache. Otherwise, they are computed by fetching the feature values from the feature
         * matrix and applying a binning method.
         */
        class Callback : public IBinningObserver, public IRuleRefinementCallback<BinVector> {

            private:

                ThresholdsSubset& thresholdsSubset_;

                uint32 featureIndex_;

                std::unique_ptr<IStatistics::IHistogramBuilder> histogramBuilderPtr_;

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
                        thresholdsSubset_.thresholds_.featureMatrixPtr_->fetchFeatureVector(featureIndex_,
                                                                                            featureVectorPtr);
                        uint32 numBins = thresholdsSubset_.thresholds_.binningPtr_->getNumBins(*featureVectorPtr);
                        binCacheEntry.binVectorPtr =  std::move(std::make_unique<BinVector>(numBins));
                        histogramBuilderPtr_ = thresholdsSubset_.thresholds_.statisticsPtr_->buildHistogram(numBins);
                        currentBinVector_ = binCacheEntry.binVectorPtr.get();
                        thresholdsSubset_.thresholds_.binningPtr_->createBins(numBins, *featureVectorPtr, *this);
                        binCacheEntry.histogramPtr = std::move(histogramBuilderPtr_->build());
                    }

                    return std::make_unique<Result>(*binCacheEntry.histogramPtr, *binCacheEntry.binVectorPtr);
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

                    IndexedValue<float32> example;
                    example.index = entry.index;
                    example.value = entry.value;
                    currentBinVector_->addExample(binIndex, example);

                    histogramBuilderPtr_->onBinUpdate(binIndex, entry);
                }

        };

        ApproximateThresholds& thresholds_;

        template<class T>
        std::unique_ptr<IRuleRefinement> createApproximateRuleRefinement(const T& labelIndices, uint32 featureIndex) {
            thresholds_.cache_.emplace(featureIndex, BinCacheEntry());
            std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex);
            std::unique_ptr<IHeadRefinement> headRefinementPtr =
                thresholds_.headRefinementFactoryPtr_->create(labelIndices);
            return std::make_unique<ApproximateRuleRefinement<T>>(std::move(headRefinementPtr), labelIndices,
                                                                  featureIndex, std::move(callbackPtr));
        }

        CoverageMask coverageMask_;

        uint32 numModifications_;

        std::unordered_map<uint32, FilteredCacheEntry<BinVector>> cacheFiltered_;

    public:

        /**
         * @param thresholds A reference to an object of type `ApproximateThresholds` that stores the thresholds
         */
        ThresholdsSubset(ApproximateThresholds& thresholds)
            : thresholds_(thresholds), coverageMask_(CoverageMask(thresholds.getNumExamples())) {
            numModifications_ = 0;
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
            //TODO: In this Branch
            numModifications_++;

            uint32 featureIndex = refinement.featureIndex;
            auto cacheFilteredIterator = cacheFiltered_.find(featureIndex);
            FilteredCacheEntry<BinVector>& cacheEntry = cacheFilteredIterator->second;
            BinVector* binVector = cacheEntry.vectorPtr.get();

            filterCurrentVector(*binVector, cacheEntry, refinement.start, refinement.end, refinement.covered,
                                numModifications_, coverageMask_);
        }

        void filterThresholds(const Condition& condition) override {

        }

        void resetThresholds() override {

        }

        const CoverageMask& getCoverageMask() const {
            return coverageMask_;
        }

        float64 evaluateOutOfSample(const CoverageMask& coverageMask, const AbstractPrediction& head) const override {
            return 0;
        }

        void recalculatePrediction(const CoverageMask& coverageMask, Refinement& refinement) const override {

        }

        void applyPrediction(const AbstractPrediction& prediction) override {

        }

};

ApproximateThresholds::ApproximateThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                             std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                                             std::shared_ptr<IStatistics> statisticsPtr,
                                             std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
                                             std::shared_ptr<IBinning> binningPtr)
    : AbstractThresholds(featureMatrixPtr, nominalFeatureMaskPtr, statisticsPtr, headRefinementFactoryPtr),
      binningPtr_(binningPtr) {

}

std::unique_ptr<IThresholdsSubset> ApproximateThresholds::createSubset(const IWeightVector& weights) {
    return std::make_unique<ApproximateThresholds::ThresholdsSubset>(*this);
}
