#include "thresholds_common.h"
#include "thresholds_approximate.h"
#include "rule_refinement/rule_refinement_approximate.h"


static inline void filterCurrentVector(BinVector& vector, FilteredCacheEntry<BinVector>& cacheEntry,
                                       intp conditionEnd, bool covered, uint32 numConditions,
                                       CoverageMask& coverageMask) {
    uint32 numTotalElements = vector.getNumElements();
    uint32 numElements = covered ? conditionEnd : (numTotalElements > conditionEnd ? numTotalElements - conditionEnd : 0);
    bool wasEmpty = false;

    BinVector* filteredVector = cacheEntry.vectorPtr.get();

    if (filteredVector == nullptr) {
        cacheEntry.vectorPtr = std::make_unique<BinVector>(numElements);
        filteredVector = cacheEntry.vectorPtr.get();
        wasEmpty = true;
    }

    typename BinVector::const_iterator iterator = vector.cbegin();
    BinVector::iterator filteredIterator = filteredVector->begin();
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

    for(intp r = start; r < end; r++) {
        for (BinVector::example_const_iterator it = vector.examples_cbegin(r); it != vector.examples_cend(r); it++) {
            BinVector::Example example = *it;
            coverageMaskIterator[example.index] = numConditions;

            if (wasEmpty) {
                filteredVector->addExample(r, example);
            }
        }

        filteredIterator[i].numExamples = iterator[r].numExamples;
        filteredIterator[i].minValue = iterator[r].minValue;
        filteredIterator[i].maxValue = iterator[r].maxValue;
        i++;
    }

    filteredVector->setNumElements(numElements);
    cacheEntry.numConditions = numConditions;
}

static inline void filterAnyVector(BinVector& vector, FilteredCacheEntry<BinVector>& cacheEntry,
                                   uint32 numConditions, const CoverageMask& coverageMask){
    //TODO: in this branch
    uint32 maxElements = vector.getNumElements();
    BinVector* filteredVector = cacheEntry.vectorPtr.get();

    if (filteredVector == nullptr) {
        cacheEntry.vectorPtr = std::make_unique<BinVector>(maxElements);
        filteredVector = cacheEntry.vectorPtr.get();
    }

    typename BinVector::const_iterator iterator = vector.cbegin();
    BinVector::iterator filteredIterator = filteredVector->begin();

    BinVector result(maxElements);

    for(intp r = 0; r < maxElements; r++) {
        for(BinVector::example_const_iterator it = vector.examples_cbegin(r); it != vector.examples_cend(r); it++){
            BinVector::Example example = *it;
            uint32 index = example.index;

            if (coverageMask.isCovered(index)) {

            }
        }
    }
}


/**
 * Provides access to a subset of the thresholds that are stored by an instance of the class `ApproximateThresholds`.
 */
class ApproximateThresholds::ThresholdsSubset : public IThresholdsSubset {

    private:

        /**
         * A callback that allows to retrieve bins and corresponding statistics. If available, the bins and statistics
         * are retrieved from the cache. Otherwise, they are computed by fetching the feature values from the feature
         * matrix and applying a binning method.
         */
        class Callback : public IBinningObserver<float32>, public IRuleRefinementCallback<BinVector> {

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
                //TODO: in this Branch
                    auto cacheFilteredIterator = thresholdsSubset_.cacheFiltered_.find(featureIndex_);
                    FilteredCacheEntry<BinVector>& cacheEntry = cacheFilteredIterator->second;

                    auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                    BinCacheEntry& binCacheEntry = cacheIterator->second;

                    if(cacheEntry.vectorPtr.get() == nullptr){

                        if (binCacheEntry.binVectorPtr.get() == nullptr) {
                            std::unique_ptr<FeatureVector> featureVectorPtr;
                            thresholdsSubset_.thresholds_.featureMatrixPtr_->fetchFeatureVector(featureIndex_,
                                                                                            featureVectorPtr);
                            IFeatureBinning::FeatureInfo featureInfo =
                                thresholdsSubset_.thresholds_.binningPtr_->getFeatureInfo(*featureVectorPtr);
                            uint32 numBins = featureInfo.numBins;
                            binCacheEntry.binVectorPtr = std::move(std::make_unique<BinVector>(numBins));
                            histogramBuilderPtr_ = thresholdsSubset_.thresholds_.statisticsPtr_->buildHistogram(numBins);
                            currentBinVector_ = binCacheEntry.binVectorPtr.get();
                            thresholdsSubset_.thresholds_.binningPtr_->createBins(featureInfo, *featureVectorPtr, *this);
                            binCacheEntry.histogramPtr = std::move(histogramBuilderPtr_->build());
                        }
                        cacheEntry.vectorPtr = std::move(binCacheEntry.binVectorPtr);
                        return std::make_unique<Result>(*binCacheEntry.histogramPtr, *binCacheEntry.binVectorPtr);
                    }

                    return std::make_unique<Result>(*binCacheEntry.histogramPtr, *cacheEntry.vectorPtr);
                }

                void onBinUpdate(uint32 binIndex, uint32 originalIndex, float32 value) override {
                    BinVector::iterator binIterator = currentBinVector_->begin();
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
                    currentBinVector_->addExample(binIndex, example);

                    histogramBuilderPtr_->onBinUpdate(binIndex, originalIndex, value);
                }

        };

        ApproximateThresholds& thresholds_;

        CoverageMask coverageMask_;

        uint32 numModifications_;

        std::unordered_map<uint32, FilteredCacheEntry<BinVector>> cacheFiltered_;

        template<class T>
        std::unique_ptr<IRuleRefinement> createApproximateRuleRefinement(const T& labelIndices, uint32 featureIndex) {
            auto cacheFilteredIterator = cacheFiltered_.emplace(featureIndex, FilteredCacheEntry<BinVector>()).first;
            BinVector* binVector = cacheFilteredIterator->second.vectorPtr.get();

            if (binVector == nullptr) {
                thresholds_.cache_.emplace(featureIndex, BinCacheEntry());
            }

            std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex);
            std::unique_ptr<IHeadRefinement> headRefinementPtr =
                thresholds_.headRefinementFactoryPtr_->create(labelIndices);
            return std::make_unique<ApproximateRuleRefinement<T>>(std::move(headRefinementPtr), labelIndices,
                                                                  featureIndex, std::move(callbackPtr));
        }

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
            numModifications_++;

            uint32 featureIndex = refinement.featureIndex;
            auto cacheFilteredIterator = cacheFiltered_.find(featureIndex);
            FilteredCacheEntry<BinVector>& cacheEntry = cacheFilteredIterator->second;
            BinVector* binVector = cacheEntry.vectorPtr.get();

            if (binVector == nullptr) {
                auto cacheIterator = thresholds_.cache_.find(featureIndex);
                BinCacheEntry& binCacheEntry = cacheIterator->second;
                binVector = binCacheEntry.binVectorPtr.get();
            }

            filterCurrentVector(*binVector, cacheEntry, refinement.end, refinement.covered, numModifications_,
                                coverageMask_);
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
                                             std::shared_ptr<IFeatureBinning> binningPtr)
    : AbstractThresholds(featureMatrixPtr, nominalFeatureMaskPtr, statisticsPtr, headRefinementFactoryPtr),
      binningPtr_(binningPtr) {

}

std::unique_ptr<IThresholdsSubset> ApproximateThresholds::createSubset(const IWeightVector& weights) {
    return std::make_unique<ApproximateThresholds::ThresholdsSubset>(*this);
}
