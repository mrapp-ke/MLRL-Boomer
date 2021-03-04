#include "common/binning/feature_binning_equal_width.hpp"
#include "common/binning/bin_index_vector_dense.hpp"
#include "common/binning/binning.hpp"
#include <unordered_set>
#include <tuple>


static inline std::tuple<uint32, float32, float32> preprocess(const FeatureVector& featureVector, float32 binRatio,
                                                              uint32 minBins, uint32 maxBins) {
    std::tuple<uint32, float32, float32> result;
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        float32 minValue = featureIterator[0].value;
        float32 maxValue = minValue;
        uint32 numDistinctValues = 1;
        std::unordered_set<float32> distinctValues;

        for (uint32 i = 1; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if (distinctValues.insert(currentValue).second) {
                numDistinctValues++;

                if (currentValue < minValue) {
                    minValue = currentValue;
                }

                if (currentValue > maxValue) {
                    maxValue = currentValue;
                }
            }
        }

        std::get<0>(result) =
            numDistinctValues > 1 ? calculateNumBins(numDistinctValues, binRatio, minBins, maxBins) : 0;
        std::get<1>(result) = minValue;
        std::get<2>(result) = maxValue;
    } else {
        std::get<0>(result) = 0;
    }

    return result;
}

EqualWidthFeatureBinning::EqualWidthFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
    : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

}

IFeatureBinning::Result EqualWidthFeatureBinning::createBins(FeatureVector& featureVector, uint32 numExamples) const {
    Result result;
    std::tuple<uint32, float32, float32> tuple = preprocess(featureVector, binRatio_, minBins_, maxBins_);
    uint32 numBins = std::get<0>(tuple);
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(featureVector, numBins, true);
    uint32 numElements = featureVector.getNumElements();
    result.binIndicesPtr = std::make_unique<DenseBinIndexVector>(numElements);

    if (numBins > 0) {
        IBinIndexVector& binIndices = *result.binIndicesPtr;
        ThresholdVector& thresholdVector = *result.thresholdVectorPtr;
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = thresholdVector.begin();
        float32 min = std::get<1>(tuple);
        float32 max = std::get<2>(tuple);
        float32 width = (max - min) / numBins;

        for (uint32 i = 0; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;
            uint32 binIndex = (uint32) std::floor((currentValue - min) / width);

            if (binIndex >= numBins) {
                binIndex = numBins - 1;
            }

            thresholdIterator[binIndex] = 1;
            binIndices.setBinIndex(featureIterator[i].index, binIndex);
        }

        // Remove empty bins and calculate thresholds...
        uint32 mapping[numBins];
        uint32 n = 0;

        for (uint32 i = 0; i < numBins; i++) {
            mapping[i] = n;

            if (thresholdIterator[i] > 0) {
                thresholdIterator[n] = min + ((i + 1) * width);
                n++;
            }
        }

        thresholdVector.setNumElements(n, true);

        // Adjust bin indices...
        for (uint32 i = 0; i < numElements; i++) {
            uint32 binIndex = binIndices.getBinIndex(i);
            binIndices.setBinIndex(i, mapping[binIndex]);
        }
    }

    return result;
}
