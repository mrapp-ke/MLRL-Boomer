#include "mlrl/common/input/feature_binning_equal_width.hpp"

#include "feature_type_numerical_common.hpp"
#include "feature_vector_decorator_binned.hpp"
#include "mlrl/common/data/array.hpp"
#include "mlrl/common/data/tuple.hpp"
#include "mlrl/common/util/math.hpp"
#include "mlrl/common/util/validation.hpp"

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

static inline uint32 getBinIndex(float32 value, float32 min, float32 width, uint32 numBins) {
    uint32 binIndex = static_cast<uint32>(std::floor((value - min) / width));
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
                numericalFeatureVector.sparseValue = featureMatrix.sparseValue;
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
  const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
    return std::make_unique<EqualWidthFeatureBinningFactory>(binRatio_, minBins_, maxBins_);
}
