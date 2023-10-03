#include "mlrl/common/input/feature_type_nominal.hpp"

#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/input/feature_vector_nominal.hpp"

#include <gtest/gtest.h>

TEST(NominalFeatureTypeTest, createNominalFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 8;
    float32* features = new float32[numExamples];
    features[0] = 1.0;
    features[1] = 0.0;
    features[2] = NAN;
    features[3] = 1.0;
    features[4] = 0.0;
    features[5] = NAN;
    features[6] = -1.0;
    features[7] = 0.0;
    FortranContiguousConstView<const float32> view(numExamples, 1, features);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const NominalFeatureVector* featureVector = dynamic_cast<const NominalFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getMajorityValue(), (int32) 0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 2);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(2));
    EXPECT_TRUE(featureVector->isMissing(5));

    // Check for regular feature values...
    std::unordered_set<int32> values;

    for (auto it = featureVector->values_cbegin(); it != featureVector->values_cend(); it++) {
        values.emplace(*it);
    }

    EXPECT_TRUE(values.find(-1) != values.end());
    EXPECT_TRUE(values.find(1) != values.end());

    // Check indices associated with the feature values...
    for (uint32 i = 0; i < 2; i++) {
        int32 value = featureVector->values_cbegin()[i];
        std::unordered_set<uint32> indices;

        for (auto it = featureVector->indices_cbegin(i); it != featureVector->indices_cend(i); it++) {
            indices.emplace(*it);
        }

        if (value == -1) {
            EXPECT_EQ(indices.size(), 1);
            EXPECT_TRUE(indices.find(6) != indices.end());
        } else {
            EXPECT_EQ(indices.size(), 2);
            EXPECT_TRUE(indices.find(0) != indices.end());
            EXPECT_TRUE(indices.find(3) != indices.end());
        }
    }

    delete[] features;
}

TEST(NominalFeatureTypeTest, createBinaryFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 7;
    float32* features = new float32[numExamples];
    features[0] = 1.0;
    features[1] = 0.0;
    features[2] = NAN;
    features[3] = 1.0;
    features[4] = 0.0;
    features[5] = NAN;
    features[6] = 0.0;
    FortranContiguousConstView<const float32> view(numExamples, 1, features);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const BinaryFeatureVector* featureVector = dynamic_cast<const BinaryFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getMajorityValue(), (int32) 0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 1);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(2));
    EXPECT_TRUE(featureVector->isMissing(5));

    // Check for regular feature values...
    int32 minorityValue = featureVector->values_cbegin()[0];
    EXPECT_EQ(minorityValue, (int32) 1);

    // Check indices associated with the feature values...
    std::unordered_set<uint32> indices;

    for (auto it = featureVector->indices_cbegin(0); it != featureVector->indices_cend(0); it++) {
        indices.emplace(*it);
    }

    EXPECT_EQ(indices.size(), 2);
    EXPECT_TRUE(indices.find(0) != indices.end());
    EXPECT_TRUE(indices.find(3) != indices.end());

    delete[] features;
}

TEST(NominalFeatureTypeTest, createEqualFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 2;
    float32* features = new float32[numExamples];
    features[0] = 0.0;
    features[1] = 0.0;
    FortranContiguousConstView<const float32> view(numExamples, 1, features);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    delete[] features;
}

TEST(NominalFeatureTypeTest, createNominalFeatureVectorFromDenseCscView) {
    // Initialize feature matrix...
    uint32 numDense = 8;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    data[0] = 1.0;
    rowIndices[0] = 0;
    data[1] = 0.0;
    rowIndices[1] = 1;
    data[2] = NAN;
    rowIndices[2] = 2;
    data[3] = 1.0;
    rowIndices[3] = 3;
    data[4] = 0.0;
    rowIndices[4] = 4;
    data[5] = NAN;
    rowIndices[5] = 5;
    data[6] = -1.0;
    rowIndices[6] = 6;
    data[7] = 0.0;
    rowIndices[7] = 7;
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscConstView<const float32> view(numDense, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const NominalFeatureVector* featureVector = dynamic_cast<const NominalFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getMajorityValue(), (int32) 0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 2);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(2));
    EXPECT_TRUE(featureVector->isMissing(5));

    // Check for regular feature values...
    std::unordered_set<int32> values;

    for (auto it = featureVector->values_cbegin(); it != featureVector->values_cend(); it++) {
        values.emplace(*it);
    }

    EXPECT_TRUE(values.find(-1) != values.end());
    EXPECT_TRUE(values.find(1) != values.end());

    // Check indices associated with the feature values...
    for (uint32 i = 0; i < 2; i++) {
        int32 value = featureVector->values_cbegin()[i];
        std::unordered_set<uint32> indices;

        for (auto it = featureVector->indices_cbegin(i); it != featureVector->indices_cend(i); it++) {
            indices.emplace(*it);
        }

        if (value == -1) {
            EXPECT_EQ(indices.size(), 1);
            EXPECT_TRUE(indices.find(6) != indices.end());
        } else {
            EXPECT_EQ(indices.size(), 2);
            EXPECT_TRUE(indices.find(0) != indices.end());
            EXPECT_TRUE(indices.find(3) != indices.end());
        }
    }

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}

TEST(NominalFeatureTypeTest, createBinaryFeatureVectorFromDenseCscView) {
    // Initialize feature matrix...
    uint32 numDense = 7;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    data[0] = 1.0;
    rowIndices[0] = 0;
    data[1] = 0.0;
    rowIndices[1] = 1;
    data[2] = NAN;
    rowIndices[2] = 2;
    data[3] = 1.0;
    rowIndices[3] = 3;
    data[4] = 0.0;
    rowIndices[4] = 4;
    data[5] = NAN;
    rowIndices[5] = 5;
    data[6] = 0.0;
    rowIndices[6] = 6;
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscConstView<const float32> view(numDense, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const BinaryFeatureVector* featureVector = dynamic_cast<const BinaryFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getMajorityValue(), (int32) 0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 1);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(2));
    EXPECT_TRUE(featureVector->isMissing(5));

    // Check for regular feature values...
    int32 minorityValue = featureVector->values_cbegin()[0];
    EXPECT_EQ(minorityValue, (int32) 1);

    // Check indices associated with the feature values...
    std::unordered_set<uint32> indices;

    for (auto it = featureVector->indices_cbegin(0); it != featureVector->indices_cend(0); it++) {
        indices.emplace(*it);
    }

    EXPECT_EQ(indices.size(), 2);
    EXPECT_TRUE(indices.find(0) != indices.end());
    EXPECT_TRUE(indices.find(3) != indices.end());

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}

TEST(NominalFeatureTypeTest, createEqualFeatureVectorFromDenseCscView) {
    // Initialize feature matrix...
    uint32 numDense = 2;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    data[0] = 0.0;
    rowIndices[0] = 0;
    data[1] = 0.0;
    rowIndices[1] = 1;
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscConstView<const float32> view(numDense, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}

TEST(NominalFeatureTypeTest, createNominalFeatureVectorFromCscView) {
    // Initialize feature matrix...
    uint32 numDense = 5;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    data[0] = 1.0;
    rowIndices[0] = 0;
    data[1] = NAN;
    rowIndices[1] = 2;
    data[2] = 1.0;
    rowIndices[2] = 3;
    data[3] = NAN;
    rowIndices[3] = 5;
    data[4] = -1.0;
    rowIndices[4] = 6;
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscConstView<const float32> view(numDense + 3, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const NominalFeatureVector* featureVector = dynamic_cast<const NominalFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getMajorityValue(), (int32) 0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 2);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(2));
    EXPECT_TRUE(featureVector->isMissing(5));

    // Check for regular feature values...
    std::unordered_set<int32> values;

    for (auto it = featureVector->values_cbegin(); it != featureVector->values_cend(); it++) {
        values.emplace(*it);
    }

    EXPECT_TRUE(values.find(-1) != values.end());
    EXPECT_TRUE(values.find(1) != values.end());

    // Check indices associated with the feature values...
    for (uint32 i = 0; i < 2; i++) {
        int32 value = featureVector->values_cbegin()[i];
        std::unordered_set<uint32> indices;

        for (auto it = featureVector->indices_cbegin(i); it != featureVector->indices_cend(i); it++) {
            indices.emplace(*it);
        }

        if (value == -1) {
            EXPECT_EQ(indices.size(), 1);
            EXPECT_TRUE(indices.find(6) != indices.end());
        } else {
            EXPECT_EQ(indices.size(), 2);
            EXPECT_TRUE(indices.find(0) != indices.end());
            EXPECT_TRUE(indices.find(3) != indices.end());
        }
    }

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}

TEST(NominalFeatureTypeTest, createBinaryFeatureVectorFromCscView) {
    // Initialize feature matrix...
    uint32 numDense = 4;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    data[0] = 1.0;
    rowIndices[0] = 0;
    data[1] = NAN;
    rowIndices[1] = 2;
    data[2] = 1.0;
    rowIndices[2] = 3;
    data[3] = NAN;
    rowIndices[3] = 5;
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscConstView<const float32> view(numDense + 3, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const BinaryFeatureVector* featureVector = dynamic_cast<const BinaryFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    // Check dimensionality of feature vector...
    EXPECT_FLOAT_EQ(featureVector->getMajorityValue(), (int32) 0);
    EXPECT_EQ(featureVector->getNumElements(), (uint32) 1);

    // Check for missing feature values....
    EXPECT_TRUE(featureVector->isMissing(2));
    EXPECT_TRUE(featureVector->isMissing(5));

    // Check for regular feature values...
    int32 minorityValue = featureVector->values_cbegin()[0];
    EXPECT_EQ(minorityValue, (int32) 1);

    // Check indices associated with the feature values...
    std::unordered_set<uint32> indices;

    for (auto it = featureVector->indices_cbegin(0); it != featureVector->indices_cend(0); it++) {
        indices.emplace(*it);
    }

    EXPECT_EQ(indices.size(), 2);
    EXPECT_TRUE(indices.find(0) != indices.end());
    EXPECT_TRUE(indices.find(3) != indices.end());

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}

TEST(NominalFeatureTypeTest, createEqualFeatureVectorFromCscView) {
    // Initialize feature matrix...
    uint32 numDense = 0;
    float32* data = new float32[numDense];
    uint32* rowIndices = new uint32[numDense];
    uint32* indptr = new uint32[2];
    indptr[0] = 0;
    indptr[1] = numDense;
    CscConstView<const float32> view(numDense + 3, 1, data, rowIndices, indptr);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = NominalFeatureType().createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}
