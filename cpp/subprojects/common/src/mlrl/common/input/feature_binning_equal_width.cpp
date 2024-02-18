#include "mlrl/common/input/feature_binning_equal_width.hpp"

#include "feature_binning_nominal.hpp"
#include "feature_type_numerical_common.hpp"
#include "feature_vector_decorator_binned.hpp"
#include "mlrl/common/binning/bin_index_vector_dense.hpp"
#include "mlrl/common/binning/bin_index_vector_dok.hpp"
#include "mlrl/common/data/array.hpp"
#include "mlrl/common/data/tuple.hpp"
#include "mlrl/common/util/math.hpp"
#include "mlrl/common/util/validation.hpp"

#include <tuple>
#include <unordered_set>

static inline Tuple<float32> getMinAndMaxFeatureValue(const NumericalFeatureVector& numericalFeatureVector) {
    uint32 numElements = numericalFeatureVector.numElements;
    float32 min;
    uint32 i;

    if (numericalFeatureVector.sparse) {
        min = numericalFeatureVector.sparseValue;
        i = 0;
    } else {
        min = numericalFeatureVector[0].value;
        i = 1;
    }

    float32 max = min;

    for (; i < numElements; i++) {
        float32 currentValue = numericalFeatureVector[i].value;

        if (currentValue < min) {
            min = currentValue;
        } else if (currentValue > max) {
            max = currentValue;
        }
    }

    return Tuple<float32>(min, max);
}

static inline constexpr uint32 getBinIndex(float32 value, float32 min, float32 width, uint32 numBins) {
    uint32 binIndex = (uint32) std::floor((value - min) / width);
    return binIndex >= numBins ? numBins - 1 : binIndex;
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  AllocatedMissingFeatureVector&& missingFeatureVector, const NumericalFeatureVector& numericalFeatureVector,
  uint32 numExamples, float32 binRatio, uint32 minBins, uint32 maxBins) {
    uint32 numWidths = calculateBoundedFraction(numExamples, binRatio, minBins, maxBins);

    if (numWidths > 0) {
        const Tuple<float32> tuple = getMinAndMaxFeatureValue(numericalFeatureVector);
        float32 min = tuple.first;
        float32 max = tuple.second;
        float32 width = (max - min) / numWidths;
        uint32 numElements = numericalFeatureVector.numElements;
        float32 sparseValue = numericalFeatureVector.sparseValue;
        uint32 sparseBinIndex = getBinIndex(sparseValue, min, width, numWidths);
        AllocatedBinnedFeatureVector binnedFeatureVector(numWidths, numElements, sparseBinIndex);
        AllocatedBinnedFeatureVector::threshold_iterator thresholdIterator = binnedFeatureVector.thresholds_begin();
        AllocatedBinnedFeatureVector::index_iterator indptrIterator = binnedFeatureVector.indptr;

        // Iterate all non-sparse feature values and determine the bins they should be assigned to...
        Array<uint32> numExamplesPerBin(numWidths, true);

        for (uint32 i = 0; i < numElements; i++) {
            float32 currentValue = numericalFeatureVector[i].value;
            uint32 binIndex = getBinIndex(currentValue, min, width, numWidths);

            if (binIndex != sparseBinIndex) {
                numExamplesPerBin[binIndex]++;
            }
        }

        // Remove empty bins and calculate thresholds...
        Array<uint32> mapping(numWidths);
        uint32 numIndices = 0;
        uint32 numBins = 0;

        for (uint32 i = 0; i < numWidths; i++) {
            uint32 numExamplesInCurrentBin = numExamplesPerBin[i];

            if (i == sparseBinIndex || numExamplesInCurrentBin > 0) {
                thresholdIterator[numBins] = min + ((numBins + 1) * width);
                indptrIterator[numBins] = numIndices;

                if (i != sparseBinIndex) {
                    numIndices += numExamplesInCurrentBin;
                } else {
                    binnedFeatureVector.sparseBinIndex = numBins;
                }

                mapping[i] = numBins;
                numBins++;
            } else {
                mapping[i] = sparseBinIndex;
            }
        }

        // Set the indices of the examples that have been assigned to each bin...
        for (uint32 i = 0; i < numElements; i++) {
            const IndexedValue<float32>& entry = numericalFeatureVector[i];
            float32 currentValue = entry.value;
            uint32 originalBinIndex = getBinIndex(currentValue, min, width, numWidths);

            if (originalBinIndex != sparseBinIndex) {
                uint32 binIndex = mapping[originalBinIndex];

                if (binIndex != binnedFeatureVector.sparseBinIndex) {
                    uint32 numExamplesInCurrentBin = numExamplesPerBin[originalBinIndex];
                    uint32 numRemaining = numExamplesInCurrentBin - 1;
                    numExamplesPerBin[originalBinIndex] = numRemaining;
                    BinnedFeatureVector::index_iterator indexIterator = binnedFeatureVector.indices_begin(binIndex);
                    indexIterator[numRemaining] = entry.index;
                }
            }
        }

        if (numBins > 1) {
            binnedFeatureVector.resize(numBins, numIndices);
            return std::make_unique<BinnedFeatureVectorDecorator>(std::move(binnedFeatureVector),
                                                                  std::move(missingFeatureVector));
        }
    }

    return std::make_unique<EqualFeatureVector>();
}

static inline std::tuple<uint32, float32, float32> preprocess(const FeatureVector& featureVector, bool sparse,
                                                              float32 binRatio, uint32 minBins, uint32 maxBins) {
    std::tuple<uint32, float32, float32> result;
    uint32 numElements = featureVector.getNumElements();

    if (numElements > 0) {
        FeatureVector::const_iterator featureIterator = featureVector.cbegin();
        float32 minValue;
        uint32 i;

        if (sparse) {
            minValue = 0;
            i = 0;
        } else {
            minValue = featureIterator[0].value;
            i = 1;
        }

        float32 maxValue = minValue;
        uint32 numDistinctValues = 1;
        std::unordered_set<float32> distinctValues;

        for (; i < numElements; i++) {
            float32 currentValue = featureIterator[i].value;

            if ((!sparse || currentValue != 0) && distinctValues.insert(currentValue).second) {
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
          numDistinctValues > 1 ? calculateBoundedFraction(numDistinctValues, binRatio, minBins, maxBins) : 0;
        std::get<1>(result) = minValue;
        std::get<2>(result) = maxValue;
    } else {
        std::get<0>(result) = 0;
    }

    return result;
}

/**
 * An implementation of the type `IFeatureBinning` that assigns numerical feature values to bins, such that each bin
 * contains values from equally sized value ranges.
 */
class EqualWidthFeatureBinning final : public IFeatureBinning {
    private:

        const float32 binRatio_;

        const uint32 minBins_;

        const uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualWidthFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
            : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

        Result createBins(FeatureVector& featureVector, uint32 numExamples) const override {
            Result result;
            uint32 numElements = featureVector.getNumElements();
            bool sparse = numElements < numExamples;
            std::tuple<uint32, float32, float32> tuple =
              preprocess(featureVector, sparse, binRatio_, minBins_, maxBins_);
            uint32 numBins = std::get<0>(tuple);
            result.thresholdVectorPtr = std::make_unique<ThresholdVector>(featureVector, numBins, true);

            if (sparse) {
                result.binIndicesPtr = std::make_unique<DokBinIndexVector>();
            } else {
                result.binIndicesPtr = std::make_unique<DenseBinIndexVector>(numElements);
            }

            if (numBins > 0) {
                IBinIndexVector& binIndices = *result.binIndicesPtr;
                ThresholdVector& thresholdVector = *result.thresholdVectorPtr;
                FeatureVector::const_iterator featureIterator = featureVector.cbegin();
                ThresholdVector::iterator thresholdIterator = thresholdVector.begin();
                float32 min = std::get<1>(tuple);
                float32 max = std::get<2>(tuple);
                float32 width = (max - min) / numBins;
                uint32 sparseBinIndex;

                // If there are any sparse values, identify the bin they belong to...
                if (sparse) {
                    sparseBinIndex = getBinIndex(0, min, width, numBins);
                    thresholdIterator[sparseBinIndex] = 1;
                    thresholdVector.setSparseBinIndex(sparseBinIndex);
                } else {
                    sparseBinIndex = numBins;
                }

                // Iterate all non-sparse feature values and identify the bins they belong to...
                for (uint32 i = 0; i < numElements; i++) {
                    float32 currentValue = featureIterator[i].value;

                    if (!sparse || currentValue != 0) {
                        uint32 binIndex = getBinIndex(currentValue, min, width, numBins);

                        if (binIndex != sparseBinIndex) {
                            thresholdIterator[binIndex] = 1;
                            binIndices.setBinIndex(featureIterator[i].index, binIndex);
                        }
                    }
                }

                // Remove empty bins and calculate thresholds...
                Array<uint32> mapping(numBins);
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
                DokBinIndexVector* dokBinIndices = dynamic_cast<DokBinIndexVector*>(&binIndices);

                if (dokBinIndices) {
                    for (auto it = dokBinIndices->begin(); it != dokBinIndices->end(); it++) {
                        uint32 binIndex = it->second;
                        it->second = mapping[binIndex];
                    }
                } else {
                    for (uint32 i = 0; i < numElements; i++) {
                        uint32 binIndex = binIndices.getBinIndex(i);
                        binIndices.setBinIndex(i, mapping[binIndex]);
                    }
                }
            }

            return result;
        }

        bool isOrdinal() const override {
            return false;
        }

        bool isNominal() const override {
            return false;
        }

        std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) const override {
            // Create a numerical feature vector from the given feature matrix...
            const std::unique_ptr<NumericalFeatureVectorDecorator> featureVectorDecoratorPtr =
              createNumericalFeatureVector(featureIndex, featureMatrix);

            // Check if all feature values are equal...
            const NumericalFeatureVector& numericalFeatureVector = featureVectorDecoratorPtr->getView().firstView;
            uint32 numElements = numericalFeatureVector.numElements;

            if (numElements > 0
                && !isEqual(numericalFeatureVector[0].value, numericalFeatureVector[numElements - 1].value)) {
                return createFeatureVectorInternally(std::move(featureVectorDecoratorPtr->getView().secondView),
                                                     numericalFeatureVector, featureMatrix.numRows, binRatio_, minBins_,
                                                     maxBins_);
            }

            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const CscView<const float32>& featureMatrix) const override {
            // Create a numerical feature vector from the given feature matrix...
            const std::unique_ptr<NumericalFeatureVectorDecorator> featureVectorDecoratorPtr =
              createNumericalFeatureVector(featureIndex, featureMatrix);

            // Check if all feature values are equal...
            NumericalFeatureVector& numericalFeatureVector = featureVectorDecoratorPtr->getView().firstView;
            uint32 numElements = numericalFeatureVector.numElements;
            uint32 numExamples = featureMatrix.numRows;

            if (numElements > 0
                && (numElements < numExamples
                    || !isEqual(numericalFeatureVector[0].value, numericalFeatureVector[numElements - 1].value))) {
                numericalFeatureVector.sparse = numElements < numExamples;
                return createFeatureVectorInternally(std::move(featureVectorDecoratorPtr->getView().secondView),
                                                     numericalFeatureVector, numExamples, binRatio_, minBins_,
                                                     maxBins_);
            }

            return std::make_unique<EqualFeatureVector>();
        }
};

/**
 * Allows to create instances of the type `IFeatureBinning` that assign numerical feature values to bins, such that each
 * bin contains values from equally sized value ranges.
 */
class EqualWidthFeatureBinningFactory final : public IFeatureBinningFactory {
    private:

        const float32 binRatio_;

        const uint32 minBins_;

        const uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualWidthFeatureBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
            : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

        std::unique_ptr<IFeatureBinning> create() const override {
            return std::make_unique<EqualWidthFeatureBinning>(binRatio_, minBins_, maxBins_);
        }
};

EqualWidthFeatureBinningConfig::EqualWidthFeatureBinningConfig() : binRatio_(0.33f), minBins_(2), maxBins_(0) {}

float32 EqualWidthFeatureBinningConfig::getBinRatio() const {
    return binRatio_;
}

IEqualWidthFeatureBinningConfig& EqualWidthFeatureBinningConfig::setBinRatio(float32 binRatio) {
    assertGreater<float32>("binRatio", binRatio, 0);
    assertLess<float32>("binRatio", binRatio, 1);
    binRatio_ = binRatio;
    return *this;
}

uint32 EqualWidthFeatureBinningConfig::getMinBins() const {
    return minBins_;
}

IEqualWidthFeatureBinningConfig& EqualWidthFeatureBinningConfig::setMinBins(uint32 minBins) {
    assertGreaterOrEqual<uint32>("minBins", minBins, 2);
    minBins_ = minBins;
    return *this;
}

uint32 EqualWidthFeatureBinningConfig::getMaxBins() const {
    return maxBins_;
}

IEqualWidthFeatureBinningConfig& EqualWidthFeatureBinningConfig::setMaxBins(uint32 maxBins) {
    if (maxBins != 0) assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins_);
    maxBins_ = maxBins;
    return *this;
}

std::unique_ptr<IFeatureBinningFactory> EqualWidthFeatureBinningConfig::createFeatureBinningFactory(
  const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    return std::make_unique<EqualWidthFeatureBinningFactory>(binRatio_, minBins_, maxBins_);
}
