#include "mlrl/common/input/feature_type_numerical.hpp"

#include "feature_type_common.hpp"
#include "feature_vector_numerical_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/iterator/index_iterator.hpp"

#include <algorithm>

/**
 * Provides random read and write access, as well as read and write access via iterators, to the indices and values of
 * training examples stored in an `AllocatedNumericalFeatureVector`.
 */
class NumericalFeatureVectorDecorator final : public AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNumericalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        NumericalFeatureVectorDecorator(AllocatedNumericalFeatureVector&& firstView,
                                        AllocatedMissingFeatureVector&& secondView)
            : AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector>(std::move(firstView),
                                                                              std::move(secondView)) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end) const override {
            // TODO Implement
            return nullptr;
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            // TODO Implement
            return nullptr;
        }
};

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<NumericalFeatureVectorDecorator> createNumericalFeatureVector(IndexIterator indexIterator,
                                                                                            ValueIterator valueIterator,
                                                                                            uint32 numElements) {
    AllocatedNumericalFeatureVector numericalFeatureVector(numElements);
    AllocatedMissingFeatureVector missingFeatureVector;
    uint32 n = 0;

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];
        float32 value = valueIterator[i];

        if (std::isnan(value)) {
            missingFeatureVector.set(index, true);
        } else {
            IndexedValue<float32>& entry = numericalFeatureVector[n];
            entry.index = index;
            entry.value = value;
            n++;
        }
    }

    numericalFeatureVector.resize(n, true);
    return std::make_unique<NumericalFeatureVectorDecorator>(std::move(numericalFeatureVector),
                                                             std::move(missingFeatureVector));
}

static inline std::unique_ptr<NumericalFeatureVectorDecorator> createNumericalFeatureVector(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) {
    FortranContiguousView<const float32>::value_const_iterator valueIterator =
      featureMatrix.values_cbegin(featureIndex);
    uint32 numElements = featureMatrix.numRows;
    return createNumericalFeatureVector(IndexIterator(), valueIterator, numElements);
}

static inline std::unique_ptr<NumericalFeatureVectorDecorator> createNumericalFeatureVector(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) {
    CscView<const float32>::index_const_iterator indexIterator = featureMatrix.indices_cbegin(featureIndex);
    CscView<const float32>::index_const_iterator indicesEnd = featureMatrix.indices_cend(featureIndex);
    CscView<const float32>::value_const_iterator valueIterator = featureMatrix.values_cbegin(featureIndex);
    uint32 numElements = indicesEnd - indexIterator;
    return createNumericalFeatureVector(indexIterator, valueIterator, numElements);
}

template<typename FeatureMatrix>
static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(uint32 featureIndex,
                                                                            const FeatureMatrix& featureMatrix) {
    std::unique_ptr<NumericalFeatureVectorDecorator> featureVectorPtr =
      createNumericalFeatureVector(featureIndex, featureMatrix);
    NumericalFeatureVector& numericalFeatureVector = featureVectorPtr->getView().firstView;

    // Sort the feature values...
    std::sort(numericalFeatureVector.begin(), numericalFeatureVector.end(), IndexedValue<float32>::CompareValue());

    // Check if all feature values are equal...
    float32 minValue = numericalFeatureVector[0].value;
    float32 maxValue = numericalFeatureVector[numericalFeatureVector.numElements - 1].value;

    if (isEqual(minValue, maxValue)) {
        return std::make_unique<EqualFeatureVector>();
    }

    return featureVectorPtr;
}

bool NumericalFeatureType::isOrdinal() const {
    return false;
}

bool NumericalFeatureType::isNominal() const {
    return false;
}

std::unique_ptr<IFeatureVector> NumericalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}

std::unique_ptr<IFeatureVector> NumericalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}
