#include "mlrl/common/input/feature_type_numerical.hpp"

#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/input/feature_vector_numerical.hpp"

#include <cmath>

#include <gtest/gtest.h>

TEST(NumericalFeatureTypeTest, createNumericalFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 7;
    float32* features = new float32[numExamples];
    features[0] = 0.2;
    features[1] = -0.1;
    features[2] = NAN;
    features[3] = -0.2;
    features[4] = 0.0;
    features[5] = NAN;
    features[6] = 0.1;
    FortranContiguousConstView<const float32> view(numExamples, 1, features);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const NumericalFeatureVector* featureVector = dynamic_cast<const NumericalFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getSparseValue(), 0.0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 5);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(2));
    EXPECT_TRUE(featureVector->isMissing(5));

    // Check if regular feature values are sorted...
    NumericalFeatureVector::const_iterator iterator = featureVector->cbegin();
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

    delete[] features;
}

TEST(NumericalFeatureTypeTest, createEqualFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 2;
    float32* features = new float32[numExamples];
    features[0] = 0.0;
    features[1] = 0.0;
    FortranContiguousConstView<const float32> view(numExamples, 1, features);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    delete[] features;
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
    CscConstView<const float32> view(numDense + 3, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const NumericalFeatureVector* featureVector = dynamic_cast<const NumericalFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getSparseValue(), 0.0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 5);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(3));
    EXPECT_TRUE(featureVector->isMissing(7));

    // Check if regular feature values are sorted...
    NumericalFeatureVector::const_iterator iterator = featureVector->cbegin();
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
    CscConstView<const float32> view(numDense + 3, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NumericalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}