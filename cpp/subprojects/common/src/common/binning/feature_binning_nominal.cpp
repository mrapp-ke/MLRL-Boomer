#include "common/binning/feature_binning_nominal.hpp"
#include "common/binning/bin_index_vector_dense.hpp"
#include "common/binning/bin_index_vector_dok.hpp"
#include <unordered_map>


IFeatureBinning::Result NominalFeatureBinning::createBins(FeatureVector& featureVector, uint32 numExamples) const {
    Result result;
    uint32 numElements = featureVector.getNumElements();
    result.thresholdVectorPtr = std::make_unique<ThresholdVector>(featureVector, numElements);
    bool sparse = numElements < numExamples;

    if (sparse) {
        result.binIndicesPtr = std::make_unique<DokBinIndexVector>();
    } else {
        result.binIndicesPtr = std::make_unique<DenseBinIndexVector>(numExamples);
    }

    if (numElements > 0) {
        IBinIndexVector& binIndices = *result.binIndicesPtr;
        ThresholdVector& thresholdVector = *result.thresholdVectorPtr;
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        ThresholdVector::iterator thresholdIterator = thresholdVector.begin();
        std::unordered_map<float32, uint32> mapping;
        uint32 nextBinIndex = 0;

        for (uint32 i = 0; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if (!sparse || currentValue != 0) {
                uint32 index = featureIterator[i].index;
                auto mapIterator = mapping.emplace(currentValue, nextBinIndex);

                if (mapIterator.second) {
                    thresholdIterator[nextBinIndex] = currentValue;
                    binIndices.setBinIndex(index, nextBinIndex);
                    nextBinIndex++;
                } else {
                    binIndices.setBinIndex(index, mapIterator.first->second);
                }
            }
        }

        thresholdVector.setNumElements(nextBinIndex, true);
    }

    return result;
}
