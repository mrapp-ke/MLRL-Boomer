#include "mlrl/common/input/feature_type_numerical.hpp"

#include "mlrl/common/input/feature_type_common.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/input/feature_vector_numerical.hpp"
#include "mlrl/common/input/feature_vector_numerical_allocated.hpp"

#include <cmath>

#include <gtest/gtest.h>

TEST(NumericalFeatureTypeTest, createNumericalFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 7;
    AllocatedFortranContiguousView<float32> featureView(numExamples, 1);
    AllocatedFortranContiguousView<float32>::value_iterator features = featureView.values_begin(0);
    features[0] = 0.2;
    features[1] = -0.1;
    features[2] = NAN;
    features[3] = -0.2;
    features[4] = 0.0;
    features[5] = NAN;
    features[6] = 0.1;
    FortranContiguousView<const float32> view(features, numExamples, 1);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector>* featureVectorDecorator =
      dynamic_cast<const AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector>*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVectorDecorator != nullptr);

    if (featureVectorDecorator) {
        // Check for missing feature values...
        const MissingFeatureVector& missingFeatureVector = featureVectorDecorator->getView().secondView;
        EXPECT_TRUE(missingFeatureVector[2]);
        EXPECT_TRUE(missingFeatureVector[5]);

        // Check dimensionality of feature vector...
        const NumericalFeatureVector& featureVector = featureVectorDecorator->getView().firstView;
        EXPECT_FLOAT_EQ(featureVector.sparseValue, 0.0);
        EXPECT_EQ(featureVector.numElements, (uint32) 5);

        // Check if regular feature values are sorted...
        NumericalFeatureVector::const_iterator iterator = featureVector.cbegin();
        EXPECT_FLOAT_EQ(iterator[0].value, -0.2);
        EXPECT_EQ(iterator[0].index, (uint32) 3);
        EXPECT_FLOAT_EQ(iterator[1].value, -0.1);
        EXPECT_EQ(iterator[1].index, (uint32) 1);
        EXPECT_FLOAT_EQ(iterator[2].value, 0.0);
        EXPECT_EQ(iterator[2].index, (uint32) 4);
        EXPECT_FLOAT_EQ(iterator[3].value, 0.1);
        EXPECT_EQ(iterator[3].index, (uint32) 6);
        EXPECT_FLOAT_EQ(iterator[4].value, 0.2);
        EXPECT_EQ(iterator[4].index, (uint32) 0);
    }
}

TEST(NumericalFeatureTypeTest, createEqualFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 2;
    AllocatedFortranContiguousView<float32> featureView(numExamples, 1);
    AllocatedFortranContiguousView<float32>::value_iterator features = featureView.values_begin(0);
    features[0] = 0.0;
    features[1] = 0.0;
    FortranContiguousView<const float32> view(features, numExamples, 1);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);
}

TEST(NumericalFeatureTypeTest, createNumericalFeatureVectorFromCscView) {
    // Initialize feature matrix...
    uint32 numDense = 7;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    data[0] = 0.2;
    rowIndices[0] = 0;
    data[1] = -0.1;
    rowIndices[1] = 2;
    data[2] = NAN;
    rowIndices[2] = 3;
    data[3] = -0.2;
    rowIndices[3] = 5;
    data[4] = 0.0;
    rowIndices[4] = 6;
    data[5] = NAN;
    rowIndices[5] = 7;
    data[6] = 0.1;
    rowIndices[6] = 9;
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscView<const float32> view(data, rowIndices, indptr, numDense + 3, 1);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector>* featureVectorDecorator =
      dynamic_cast<const AbstractFeatureVectorDecorator<AllocatedNumericalFeatureVector>*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVectorDecorator != nullptr);

    if (featureVectorDecorator) {
        // Check for missing feature values...
        const MissingFeatureVector& missingFeatureVector = featureVectorDecorator->getView().secondView;
        EXPECT_TRUE(missingFeatureVector[3]);
        EXPECT_TRUE(missingFeatureVector[7]);

        // Check dimensionality of feature vector...
        const NumericalFeatureVector& featureVector = featureVectorDecorator->getView().firstView;
        EXPECT_FLOAT_EQ(featureVector.sparseValue, 0.0);
        EXPECT_EQ(featureVector.numElements, (uint32) 5);

        // Check if regular feature values are sorted...
        NumericalFeatureVector::const_iterator iterator = featureVector.cbegin();
        EXPECT_FLOAT_EQ(iterator[0].value, -0.2);
        EXPECT_EQ(iterator[0].index, (uint32) 5);
        EXPECT_FLOAT_EQ(iterator[1].value, -0.1);
        EXPECT_EQ(iterator[1].index, (uint32) 2);
        EXPECT_FLOAT_EQ(iterator[2].value, 0.0);
        EXPECT_EQ(iterator[2].index, (uint32) 6);
        EXPECT_FLOAT_EQ(iterator[3].value, 0.1);
        EXPECT_EQ(iterator[3].index, (uint32) 9);
        EXPECT_FLOAT_EQ(iterator[4].value, 0.2);
        EXPECT_EQ(iterator[4].index, (uint32) 0);
    }

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}

TEST(NumericalFeatureTypeTest, createEqualFeatureVectorFromCscView) {
    // Initialize feature matrix...
    uint32 numDense = 2;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    data[0] = 0.1;
    rowIndices[0] = 0;
    data[1] = 0.1;
    rowIndices[1] = 1;
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscView<const float32> view(data, rowIndices, indptr, numDense + 3, 1);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}
