#include "mlrl/common/input/feature_vector_binary.hpp"

#include "mlrl/common/input/feature_vector_equal.hpp"

#include <gtest/gtest.h>

TEST(BinaryFeatureVectorTest, createFilteredFeatureVectorFromIndices) {
    BinaryFeatureVector featureVector(10, 0, 1);
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = featureVector.createFilteredFeatureVector(existing, 0, 1);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}

TEST(BinaryFeatureVectorTest, createFilteredFeatureVectorFromCoverageMask) {
    uint32 numMinorityExamples = 10;
    BinaryFeatureVector featureVector(numMinorityExamples, 0, 1);
    BinaryFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    uint32 numMissingIndices = 10;

    for (uint32 i = numMinorityExamples; i < numMinorityExamples + numMissingIndices; i++) {
        featureVector.setMissing(i, true);
    }

    CoverageMask coverageMask(numMinorityExamples + numMissingIndices);
    uint32 indicatorValue = 1;
    coverageMask.setIndicatorValue(indicatorValue);
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    for (uint32 i = 0; i < numMinorityExamples + numMissingIndices; i++) {
        if (i % 2 == 0) {
            coverageMaskIterator[i] = indicatorValue;
        }
    }

    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = featureVector.createFilteredFeatureVector(existing, coverageMask);
    const BinaryFeatureVector* filteredFeatureVector = dynamic_cast<const BinaryFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);

    // Check filtered indices...
    BinaryFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector->indices_cbegin(0);
    BinaryFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector->indices_cend(0);
    uint32 numIndices = indicesEnd - indicesBegin;
    EXPECT_EQ(numIndices, numMinorityExamples / 2);
    std::unordered_set<uint32> indices;

    for (auto it = indicesBegin; it != indicesEnd; it++) {
        indices.emplace(*it);
    }

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(indices.find(i) != indices.end());
        } else {
            EXPECT_TRUE(indices.find(i) == indices.end());
        }
    }

    // Check missing indices...
    for (uint32 i = numMinorityExamples; i < numMinorityExamples + numMissingIndices; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(filteredFeatureVector->isMissing(i));
        } else {
            EXPECT_FALSE(filteredFeatureVector->isMissing(i));
        }
    }
}

TEST(BinaryFeatureVectorTest, createFilteredFeatureVectorFromCoverageMaskUsingExisting) {
    uint32 numMinorityExamples = 10;
    std::unique_ptr<BinaryFeatureVector> featureVector =
      std::make_unique<BinaryFeatureVector>(numMinorityExamples, 0, 1);
    BinaryFeatureVector::index_iterator indexIterator = featureVector->indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    uint32 numMissingIndices = 10;

    for (uint32 i = numMinorityExamples; i < numMinorityExamples + numMissingIndices; i++) {
        featureVector->setMissing(i, true);
    }

    CoverageMask coverageMask(numMinorityExamples + numMissingIndices);
    uint32 indicatorValue = 1;
    coverageMask.setIndicatorValue(indicatorValue);
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    for (uint32 i = 0; i < numMinorityExamples + numMissingIndices; i++) {
        if (i % 2 == 0) {
            coverageMaskIterator[i] = indicatorValue;
        }
    }

    std::unique_ptr<IFeatureVector> existing = std::move(featureVector);
    std::unique_ptr<IFeatureVector> filtered = existing->createFilteredFeatureVector(existing, coverageMask);
    const BinaryFeatureVector* filteredFeatureVector = dynamic_cast<const BinaryFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
    EXPECT_TRUE(existing.get() == nullptr);

    // Check filtered indices...
    BinaryFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector->indices_cbegin(0);
    BinaryFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector->indices_cend(0);
    uint32 numIndices = indicesEnd - indicesBegin;
    EXPECT_EQ(numIndices, numMinorityExamples / 2);
    std::unordered_set<uint32> indices;

    for (auto it = indicesBegin; it != indicesEnd; it++) {
        indices.emplace(*it);
    }

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(indices.find(i) != indices.end());
        } else {
            EXPECT_TRUE(indices.find(i) == indices.end());
        }
    }

    // Check missing indices...
    for (uint32 i = numMinorityExamples; i < numMinorityExamples + numMissingIndices; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(filteredFeatureVector->isMissing(i));
        } else {
            EXPECT_FALSE(filteredFeatureVector->isMissing(i));
        }
    }
}

TEST(BinaryFeatureVectorTest, createFilteredFeatureVectorFromCoverageMaskReturnsEqualFeatureVector) {
    uint32 numMinorityExamples = 10;
    BinaryFeatureVector featureVector(numMinorityExamples, 0, 1);
    BinaryFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    CoverageMask coverageMask(numMinorityExamples);
    coverageMask.setIndicatorValue(1);

    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = featureVector.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}
