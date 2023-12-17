#include "mlrl/common/input/feature_type_numerical.hpp"

#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/input/feature_vector_numerical.hpp"
#include "mlrl/common/iterator/index_iterator.hpp"

#include <algorithm>

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<NumericalFeatureVector> createNumericalFeatureVector(IndexIterator indexIterator,
                                                                                   ValueIterator valueIterator,
                                                                                   uint32 numElements) {
    std::unique_ptr<NumericalFeatureVector> featureVectorPtr = std::make_unique<NumericalFeatureVector>(numElements, 0);
    NumericalFeatureVector::iterator vectorIterator = featureVectorPtr->begin();
    uint32 n = 0;

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];
        float32 value = valueIterator[i];

        if (std::isnan(value)) {
            featureVectorPtr->setMissing(index, true);
        } else {
            IndexedValue<float32>& entry = vectorIterator[n];
            entry.index = index;
            entry.value = value;
            n++;
        }
    }

    featureVectorPtr->setNumElements(n, true);
    return featureVectorPtr;
}

static inline std::unique_ptr<NumericalFeatureVector> createNumericalFeatureVector(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) {
    FortranContiguousView<const float32>::value_const_iterator valueIterator =
      featureMatrix.values_cbegin(featureIndex);
    uint32 numElements = featureMatrix.numRows;
    return createNumericalFeatureVector(IndexIterator(), valueIterator, numElements);
}

static inline std::unique_ptr<NumericalFeatureVector> createNumericalFeatureVector(
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
    std::unique_ptr<NumericalFeatureVector> featureVectorPtr =
      createNumericalFeatureVector(featureIndex, featureMatrix);

    // Sort the feature values...
    std::sort(featureVectorPtr->begin(), featureVectorPtr->end(), IndexedValue<float32>::CompareValue());

    // Check if all feature values are equal...
    NumericalFeatureVector::const_iterator iterator = featureVectorPtr->cbegin();
    uint32 numElements = featureVectorPtr->getNumElements();
    float32 minValue = iterator[0].value;
    float32 maxValue = iterator[numElements - 1].value;

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
