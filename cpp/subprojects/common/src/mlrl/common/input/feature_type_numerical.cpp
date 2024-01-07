#include "mlrl/common/input/feature_type_numerical.hpp"

#include "feature_vector_decorator_numerical.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/iterator/index_iterator.hpp"

#include <algorithm>

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
    std::sort(numericalFeatureVector.begin(), numericalFeatureVector.end(), IndexedValue<float32>::CompareValue());
    return std::make_unique<NumericalFeatureVectorDecorator>(std::move(numericalFeatureVector),
                                                             std::move(missingFeatureVector));
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) {
    FortranContiguousView<const float32>::value_const_iterator valueIterator =
      featureMatrix.values_cbegin(featureIndex);
    uint32 numRows = featureMatrix.numRows;
    std::unique_ptr<NumericalFeatureVectorDecorator> featureVectorDecoratorPtr =
      createNumericalFeatureVector(IndexIterator(), valueIterator, numRows);

    // Check if all feature values are equal...
    const NumericalFeatureVector& numericalFeatureVector = featureVectorDecoratorPtr->getView().firstView;
    uint32 numElements = numericalFeatureVector.numElements;

    if (numElements > 0 && !isEqual(numericalFeatureVector[0].value, numericalFeatureVector[numElements - 1].value)) {
        return featureVectorDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) {
    CscView<const float32>::index_const_iterator indexIterator = featureMatrix.indices_cbegin(featureIndex);
    CscView<const float32>::index_const_iterator indicesEnd = featureMatrix.indices_cend(featureIndex);
    CscView<const float32>::value_const_iterator valueIterator = featureMatrix.values_cbegin(featureIndex);
    uint32 numIndices = indicesEnd - indexIterator;
    std::unique_ptr<NumericalFeatureVectorDecorator> featureVectorDecoratorPtr =
      createNumericalFeatureVector(indexIterator, valueIterator, numIndices);

    // Check if all feature values are equal...
    NumericalFeatureVector& numericalFeatureVector = featureVectorDecoratorPtr->getView().firstView;
    uint32 numElements = numericalFeatureVector.numElements;

    if (numElements > 0
        && (numElements < numIndices
            || !isEqual(numericalFeatureVector[0].value, numericalFeatureVector[numElements - 1].value))) {
        numericalFeatureVector.sparse = numElements < numIndices;
        return featureVectorDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
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
