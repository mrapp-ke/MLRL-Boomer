#include "mlrl/common/input/feature_type_numerical.hpp"

#include "feature_type_numerical_common.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/iterator/iterator_index.hpp"

#include <algorithm>

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) {
    std::unique_ptr<NumericalFeatureVectorDecorator> featureVectorDecoratorPtr =
      createNumericalFeatureVector(featureIndex, featureMatrix);

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
    std::unique_ptr<NumericalFeatureVectorDecorator> featureVectorDecoratorPtr =
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
        return featureVectorDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}

std::unique_ptr<IFeatureVector> NumericalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}

std::unique_ptr<IFeatureVector> NumericalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}
