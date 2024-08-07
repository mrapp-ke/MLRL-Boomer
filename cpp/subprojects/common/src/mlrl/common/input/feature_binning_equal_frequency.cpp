#include "mlrl/common/input/feature_binning_equal_frequency.hpp"

#include "feature_type_numerical_common.hpp"
#include "feature_vector_decorator_binned.hpp"
#include "mlrl/common/util/math.hpp"
#include "mlrl/common/util/validation.hpp"

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  AllocatedMissingFeatureVector&& missingFeatureVector, const NumericalFeatureVector& numericalFeatureVector,
  uint32 numExamples, float32 binRatio, uint32 minBins, uint32 maxBins) {
    uint32 numBins = calculateBoundedFraction(numExamples, binRatio, minBins, maxBins);

    if (numBins > 1) {
        uint32 numElements = numericalFeatureVector.numElements;
        AllocatedBinnedFeatureVector binnedFeatureVector(numBins, numElements);
        AllocatedBinnedFeatureVector::threshold_iterator thresholdIterator = binnedFeatureVector.thresholds_begin();
        AllocatedBinnedFeatureVector::index_iterator indexIterator = binnedFeatureVector.indices;
        AllocatedBinnedFeatureVector::index_iterator indptrIterator = binnedFeatureVector.indptr;
        uint32 numElementsPerBin = static_cast<uint32>(std::ceil((float64) numElements / (float64) numBins));
        bool sparse = numericalFeatureVector.sparse;
        float32 sparseValue = numericalFeatureVector.sparseValue;
        float32 previousValue = sparseValue;
        uint32 numElementsInCurrentBin = 0;
        uint32 binIndex = 0;
        uint32 numIndices = 0;
        uint32 i = 0;

        // Iterate feature values `f < sparseValue`...
        for (; i < numElements; i++) {
            const IndexedValue<float32>& entry = numericalFeatureVector[i];
            float32 currentValue = entry.value;

            if (!(currentValue < sparseValue)) {
                break;
            }

            // Feature values that are equal to the previous one must not be assigned to a new bin...
            if (!isEqual(currentValue, previousValue)) {
                // Check, if the bin is fully occupied...
                if (numElementsInCurrentBin >= numElementsPerBin) {
                    thresholdIterator[binIndex] = arithmeticMean(previousValue, currentValue);
                    indptrIterator[binIndex + 1] = numIndices;
                    numElementsInCurrentBin = 0;
                    binIndex++;
                }

                previousValue = currentValue;
            }

            indexIterator[numIndices] = entry.index;
            numElementsInCurrentBin++;
            numIndices++;
        }

        // If there are any sparse values, check if they belong to the current one or the next one...
        if (sparse) {
            uint32 numSparseValues = numExamples - numElements;

            if (numElementsInCurrentBin >= numElementsPerBin) {
                // The sparse values belong to the next bin...
                thresholdIterator[binIndex] = arithmeticMean(previousValue, sparseValue);
                indptrIterator[binIndex + 1] = numIndices;
                numElementsInCurrentBin = numSparseValues;
                binIndex++;
            } else {
                // The sparse values belong to the current bin...
                numIndices -= numElementsInCurrentBin;
                numElementsInCurrentBin += numSparseValues;
            }

            // If the current bin is not fully occupied yet, the subsequent values do also belong to it...
            previousValue = sparseValue;

            // Skip feature values that are equal to the previous one...
            for (; i < numElements; i++) {
                if (!isEqual(numericalFeatureVector[i].value, previousValue)) {
                    break;
                }

                numElementsInCurrentBin++;
            }
        }

        // Set the index of the sparse bin...
        binnedFeatureVector.sparseBinIndex = binIndex;

        // Iterate feature values `f >= sparseValue`...
        for (; i < numElements; i++) {
            const IndexedValue<float32>& entry = numericalFeatureVector[i];
            float32 currentValue = entry.value;

            // Feature values that are equal to the previous one must not be assigned to a new bin...
            if (!isEqual(currentValue, previousValue)) {
                // Check, if the bin is fully occupied...
                if (numElementsInCurrentBin >= numElementsPerBin) {
                    thresholdIterator[binIndex] = arithmeticMean(previousValue, currentValue);
                    indptrIterator[binIndex + 1] = numIndices;
                    numElementsInCurrentBin = 0;
                    binIndex++;
                }

                previousValue = currentValue;
            }

            indexIterator[numIndices] = entry.index;
            numElementsInCurrentBin++;
            numIndices++;
        }

        if (binIndex > 0) {
            binnedFeatureVector.resize(binIndex + 1, numIndices);
            return std::make_unique<BinnedFeatureVectorDecorator>(std::move(binnedFeatureVector),
                                                                  std::move(missingFeatureVector));
        }
    }

    return std::make_unique<EqualFeatureVector>();
}

/**
 * An implementation of the type `IFeatureBinning` that assigns numerical feature values to bins, such that each bin
 * contains approximately the same number of values.
 */
class EqualFrequencyFeatureBinning final : public IFeatureBinning {
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
        EqualFrequencyFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
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
 * bin contains approximately the same number of values.
 */
class EqualFrequencyFeatureBinningFactory final : public IFeatureBinningFactory {
    private:

        const float32 binRatio_;

        const uint32 minBins_;

        const uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualFrequencyFeatureBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
            : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

        std::unique_ptr<IFeatureBinning> create() const override {
            return std::make_unique<EqualFrequencyFeatureBinning>(binRatio_, minBins_, maxBins_);
        }
};

EqualFrequencyFeatureBinningConfig::EqualFrequencyFeatureBinningConfig() : binRatio_(0.33f), minBins_(2), maxBins_(0) {}

float32 EqualFrequencyFeatureBinningConfig::getBinRatio() const {
    return binRatio_;
}

IEqualFrequencyFeatureBinningConfig& EqualFrequencyFeatureBinningConfig::setBinRatio(float32 binRatio) {
    assertGreater<float32>("binRatio", binRatio, 0);
    assertLess<float32>("binRatio", binRatio, 1);
    binRatio_ = binRatio;
    return *this;
}

uint32 EqualFrequencyFeatureBinningConfig::getMinBins() const {
    return minBins_;
}

IEqualFrequencyFeatureBinningConfig& EqualFrequencyFeatureBinningConfig::setMinBins(uint32 minBins) {
    assertGreaterOrEqual<uint32>("minBins", minBins, 2);
    minBins_ = minBins;
    return *this;
}

uint32 EqualFrequencyFeatureBinningConfig::getMaxBins() const {
    return maxBins_;
}

IEqualFrequencyFeatureBinningConfig& EqualFrequencyFeatureBinningConfig::setMaxBins(uint32 maxBins) {
    if (maxBins != 0) assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins_);
    maxBins_ = maxBins;
    return *this;
}

std::unique_ptr<IFeatureBinningFactory> EqualFrequencyFeatureBinningConfig::createFeatureBinningFactory(
  const IFeatureMatrix& featureMatrix, const IOutputMatrix& outputMatrix) const {
    return std::make_unique<EqualFrequencyFeatureBinningFactory>(binRatio_, minBins_, maxBins_);
}
