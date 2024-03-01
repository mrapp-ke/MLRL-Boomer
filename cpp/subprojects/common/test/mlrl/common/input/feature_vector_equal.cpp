#include "mlrl/common/input/feature_vector_equal.hpp"

#include <gtest/gtest.h>

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromIndices) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing;
    EXPECT_THROW(featureVector.createFilteredFeatureVector(existing, Interval(0, 1)), std::runtime_error);
}

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromCoverageMask) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing;
    CoverageMask coverageMask(10);
    std::unique_ptr<IFeatureVector> filteredPtr = featureVector.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filtered = dynamic_cast<const EqualFeatureVector*>(filteredPtr.get());
    EXPECT_TRUE(filtered != nullptr);
}
