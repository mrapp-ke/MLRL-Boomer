#include "mlrl/common/input/feature_binning_equal_width.hpp"

#include "mlrl/common/input/feature_binning_equal_width.cpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

#include <gtest/gtest.h>

TEST(EqualWidthFeatureBinningTest, createBinnedFeatureVectorFromFortranContiguousView) {
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
    std::unique_ptr<IFeatureVector> featureVectorPtr = EqualWidthFeatureBinning(0.4, 1, 0).createFeatureVector(0, view);

    // Check type of feature vector...
    const AbstractFeatureVectorDecorator<AllocatedBinnedFeatureVector>* featureVectorDecorator =
      dynamic_cast<const AbstractFeatureVectorDecorator<AllocatedBinnedFeatureVector>*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVectorDecorator != nullptr);

    if (featureVectorDecorator) {
        // Check for missing feature values...
        const MissingFeatureVector& missingFeatureVector = featureVectorDecorator->getView().secondView;
        EXPECT_TRUE(missingFeatureVector[2]);
        EXPECT_TRUE(missingFeatureVector[5]);

        // Check dimensionality of feature vector...
        const BinnedFeatureVector& featureVector = featureVectorDecorator->getView().firstView;
        EXPECT_EQ(featureVector.numBins, (uint32) 3);
        EXPECT_EQ(featureVector.sparseBinIndex, (uint32) 1);

        // Check thresholds and indices associated with each bin...
        BinnedFeatureVector::threshold_const_iterator thresholdIterator = featureVector.thresholds_cbegin();
        BinnedFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(0);
        BinnedFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(0);
        uint32 numIndices = indicesEnd - indexIterator;
        EXPECT_EQ(numIndices, (uint32) 2);
        EXPECT_EQ(indexIterator[0], (uint32) 1);
        EXPECT_EQ(indexIterator[1], (uint32) 3);
        EXPECT_FLOAT_EQ(thresholdIterator[0], -0.066666663f);

        indexIterator = featureVector.indices_cbegin(1);
        indicesEnd = featureVector.indices_cend(1);
        numIndices = indicesEnd - indexIterator;
        EXPECT_EQ(numIndices, (uint32) 0);
        EXPECT_FLOAT_EQ(thresholdIterator[1], 0.066666678f);

        indexIterator = featureVector.indices_cbegin(2);
        indicesEnd = featureVector.indices_cend(2);
        numIndices = indicesEnd - indexIterator;
        EXPECT_EQ(numIndices, (uint32) 2);
        EXPECT_EQ(indexIterator[0], (uint32) 0);
        EXPECT_EQ(indexIterator[1], (uint32) 6);
    }
}

TEST(EqualWidthFeatureBinningTest, createEqualFeatureVectorFromFortranContiguousView) {
    // Initialize feature matrix...
    uint32 numExamples = 1;
    AllocatedFortranContiguousView<float32> featureView(numExamples, 1);
    AllocatedFortranContiguousView<float32>::value_iterator features = featureView.values_begin(0);
    features[0] = 0.0;
    FortranContiguousView<const float32> view(features, numExamples, 1);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = EqualWidthFeatureBinning(0.4, 1, 0).createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);
}

TEST(EqualWidthFeatureBinningTest, createBinnedFeatureVectorFromCscView) {
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
    std::unique_ptr<IFeatureVector> featureVectorPtr = EqualWidthFeatureBinning(0.3, 1, 0).createFeatureVector(0, view);

    // Check type of feature vector...
    const AbstractFeatureVectorDecorator<AllocatedBinnedFeatureVector>* featureVectorDecorator =
      dynamic_cast<const AbstractFeatureVectorDecorator<AllocatedBinnedFeatureVector>*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVectorDecorator != nullptr);

    if (featureVectorDecorator) {
        // Check for missing feature values...
        const MissingFeatureVector& missingFeatureVector = featureVectorDecorator->getView().secondView;
        EXPECT_TRUE(missingFeatureVector[3]);
        EXPECT_TRUE(missingFeatureVector[7]);

        // Check dimensionality of feature vector...
        const BinnedFeatureVector& featureVector = featureVectorDecorator->getView().firstView;
        EXPECT_EQ(featureVector.numBins, (uint32) 3);
        EXPECT_EQ(featureVector.sparseBinIndex, (uint32) 1);

        // Check thresholds and indices associated with each bin...
        BinnedFeatureVector::threshold_const_iterator thresholdIterator = featureVector.thresholds_cbegin();
        BinnedFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(0);
        BinnedFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(0);
        uint32 numIndices = indicesEnd - indexIterator;
        EXPECT_EQ(numIndices, (uint32) 2);
        EXPECT_EQ(indexIterator[0], (uint32) 2);
        EXPECT_EQ(indexIterator[1], (uint32) 5);
        EXPECT_FLOAT_EQ(thresholdIterator[0], -0.066666663f);

        indexIterator = featureVector.indices_cbegin(1);
        indicesEnd = featureVector.indices_cend(1);
        numIndices = indicesEnd - indexIterator;
        EXPECT_EQ(numIndices, (uint32) 0);
        EXPECT_FLOAT_EQ(thresholdIterator[1], 0.066666678f);

        indexIterator = featureVector.indices_cbegin(2);
        indicesEnd = featureVector.indices_cend(2);
        numIndices = indicesEnd - indexIterator;
        EXPECT_EQ(numIndices, (uint32) 2);
        EXPECT_EQ(indexIterator[0], (uint32) 0);
        EXPECT_EQ(indexIterator[1], (uint32) 9);
    }

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}

TEST(EqualWidthFeatureBinningTest, createEqualFeatureVectorFromCscView) {
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
    CscView<const float32> view(data, rowIndices, indptr, numDense, 1);

    // Create feature vector...
    std::unique_ptr<IFeatureVector> featureVectorPtr = EqualWidthFeatureBinning(0.4, 1, 0).createFeatureVector(0, view);

    // Check type of feature vector...
    const EqualFeatureVector* featureVector = dynamic_cast<const EqualFeatureVector*>(featureVectorPtr.get());
    EXPECT_TRUE(featureVector != nullptr);

    delete[] data;
    delete[] rowIndices;
    delete[] indptr;
}
