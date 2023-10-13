#include "mlrl/common/input/feature_vector_equal.hpp"

#include <gtest/gtest.h>

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromIndices) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = featureVector.createFilteredFeatureVector(existing, 0, 1);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromIndicesUsingExisting) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing = std::make_unique<EqualFeatureVector>();
    std::unique_ptr<IFeatureVector> filtered = featureVector.createFilteredFeatureVector(existing, 0, 1);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
    EXPECT_TRUE(existing.get() == nullptr);
}

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromCoverageMask) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing;
    CoverageMask coverageMask(10);
    std::unique_ptr<IFeatureVector> filtered = featureVector.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromCoverageMaskUsingExisting) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing = std::make_unique<EqualFeatureVector>();
    CoverageMask coverageMask(10);
    std::unique_ptr<IFeatureVector> filtered = featureVector.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
    EXPECT_TRUE(existing.get() == nullptr);
}
