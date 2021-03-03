#include "common/binning/feature_binning_equal_frequency.hpp"
#include "common/binning/binning.hpp"
#include "common/math/math.hpp"


EqualFrequencyFeatureBinning::EqualFrequencyFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
    : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

}

IFeatureBinning::FeatureInfo EqualFrequencyFeatureBinning::getFeatureInfo(FeatureVector& featureVector) const {
    FeatureInfo featureInfo;
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        featureVector.sortByValues();
        FeatureVector::const_iterator iterator = featureVector.cbegin();
        float32 previousValue = iterator[0].value;
        uint32 numDistinctValues = 1;

        for (uint32 i = 1; i < numElements; i++) {
            float32 value = iterator[i].value;

            if (previousValue != value) {
                numDistinctValues++;
                previousValue = value;
            }
        }

        featureInfo.numBins =
            numDistinctValues > 1 ? calculateNumBins(numDistinctValues, binRatio_, minBins_, maxBins_) : 0;
    } else {
        featureInfo.numBins = 0;
    }

    return featureInfo;
}

IFeatureBinning::Result EqualFrequencyFeatureBinning::createBins(FeatureInfo featureInfo,
                                                                 const FeatureVector& featureVector,
                                                                 Callback callback) const {
    Result result;
    uint32 numBins = featureInfo.numBins;
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(numBins);
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<BinIndexVector>(numElements);

    if (numBins > 0) {
        uint32 numElementsPerBin = (uint32) std::ceil((float) numElements / (float) numBins);
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = result.thresholdVectorPtr->begin();
        BinIndexVector::iterator binIndexIterator = result.binIndicesPtr->begin();
        uint32 binIndex = 0;
        float32 previousValue = 0;

        for (uint32 i = 0; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if (currentValue != previousValue) {
                if (i / numElementsPerBin != binIndex) {
                    thresholdIterator[binIndex] = mean(previousValue, currentValue);
                    binIndex++;
                }

                thresholdIterator[binIndex] = currentValue;
                previousValue = currentValue;
            }

            binIndexIterator[featureIterator[i].index] = binIndex;
        }

        result.thresholdVectorPtr->setNumElements(binIndex + 1, true);
    }

    return result;
}
