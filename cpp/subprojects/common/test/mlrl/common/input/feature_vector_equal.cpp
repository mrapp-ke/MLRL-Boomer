#include "mlrl/common/input/feature_vector_equal.hpp"

#include <gtest/gtest.h>

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromIndices) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing;
    EXPECT_THROW(featureVector.createFilteredFeatureVector(existing, 0, 1, false), std::runtime_error);
}

TEST(EqualFeatureVectorTest, createFilteredFeatureVectorFromCoverageMask) {
    EqualFeatureVector featureVector;
    std::unique_ptr<IFeatureVector> existing;
    CoverageMask coverageMask(10);
    EXPECT_THROW(featureVector.createFilteredFeatureVector(existing, coverageMask), std::runtime_error);
}
