#include "mlrl/common/input/feature_vector_decorator_binary.hpp"

#include "mlrl/common/input/feature_vector_binary.hpp"
#include "statistics_weighted.hpp"

#include <gtest/gtest.h>

TEST(BinaryFeatureVectorDecoratorTest, updateCoverageMaskAndStatistics) {
    uint32 numMinorityExamples = 10;
    AllocatedNominalFeatureVector featureVector(1, numMinorityExamples, 1);
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    WeightedStatistics statistics;
    uint32 numExamples = numMinorityExamples + 15;

    for (uint32 i = 0; i < numExamples; i++) {
        statistics.addCoveredStatistic(i);
    }

    BinaryFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    Interval interval(0, 1);
    CoverageMask coverageMask(numExamples);
    uint32 indicatorValue = 1;
    decorator.updateCoverageMaskAndStatistics(interval, coverageMask, indicatorValue, statistics);
    EXPECT_EQ(coverageMask.indicatorValue, indicatorValue);
    const BinaryFeatureVector& binaryFeatureVector = decorator.getView().firstView;

    for (auto it = binaryFeatureVector.indices_cbegin(0); it != binaryFeatureVector.indices_cend(0); it++) {
        uint32 index = *it;
        EXPECT_TRUE(coverageMask.isCovered(index));
        EXPECT_TRUE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
    }

    for (uint32 i = numMinorityExamples; i < numExamples; i++) {
        EXPECT_FALSE(coverageMask.isCovered(i));
        EXPECT_FALSE(statistics.coveredStatistics.find(i) != statistics.coveredStatistics.end());
    }
}

TEST(BinaryFeatureVectorDecoratorTest, updateCoverageMaskAndStatisticsInverse) {
    uint32 numMinorityExamples = 10;
    AllocatedNominalFeatureVector featureVector(1, numMinorityExamples, 1);
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    AllocatedMissingFeatureVector missingFeatureVector;
    uint32 numMissingExamples = 5;

    for (uint32 i = numMinorityExamples; i < numMinorityExamples + numMissingExamples; i++) {
        missingFeatureVector.set(i, true);
    }

    WeightedStatistics statistics;
    uint32 numExamples = numMinorityExamples + numMissingExamples + 15;

    for (uint32 i = 0; i < numExamples; i++) {
        statistics.addCoveredStatistic(i);
    }

    BinaryFeatureVectorDecorator decorator(std::move(featureVector), std::move(missingFeatureVector));
    Interval interval(0, 1, true);
    CoverageMask coverageMask(numExamples);
    uint32 indicatorValue = 1;
    decorator.updateCoverageMaskAndStatistics(interval, coverageMask, indicatorValue, statistics);
    EXPECT_EQ(coverageMask.indicatorValue, (uint32) 0);
    const BinaryFeatureVector& binaryFeatureVector = decorator.getView().firstView;

    for (auto it = binaryFeatureVector.indices_cbegin(0); it != binaryFeatureVector.indices_cend(0); it++) {
        uint32 index = *it;
        EXPECT_FALSE(coverageMask.isCovered(index));
        EXPECT_FALSE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
    }

    for (uint32 i = numMinorityExamples; i < numMinorityExamples + numMissingExamples; i++) {
        EXPECT_FALSE(coverageMask.isCovered(i));
        EXPECT_FALSE(statistics.coveredStatistics.find(i) != statistics.coveredStatistics.end());
    }

    for (uint32 i = numMinorityExamples + numMissingExamples; i < numExamples; i++) {
        EXPECT_TRUE(coverageMask.isCovered(i));
        EXPECT_TRUE(statistics.coveredStatistics.find(i) != statistics.coveredStatistics.end());
    }
}

TEST(BinaryFeatureVectorDecoratorTest, createFilteredFeatureVectorFromIndices) {
    BinaryFeatureVectorDecorator decorator(AllocatedNominalFeatureVector(1, 0, 1), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, Interval(0, 1));
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}

TEST(BinaryFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMask) {
    uint32 numMinorityExamples = 10;
    AllocatedNominalFeatureVector featureVector(1, numMinorityExamples, 1);
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    AllocatedMissingFeatureVector missingFeatureVector;
    uint32 numMissingIndices = 10;
    uint32 numExamples = numMinorityExamples + numMissingIndices;

    for (uint32 i = numMinorityExamples; i < numExamples; i++) {
        missingFeatureVector.set(i, true);
    }

    CoverageMask coverageMask(numExamples);
    uint32 indicatorValue = 1;
    coverageMask.indicatorValue = indicatorValue;
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    for (uint32 i = 0; i < numExamples; i++) {
        if (i % 2 == 0) {
            coverageMaskIterator[i] = indicatorValue;
        }
    }

    BinaryFeatureVectorDecorator decorator(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const BinaryFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const BinaryFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const BinaryFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        BinaryFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(0);
        BinaryFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(0);
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
        const MissingFeatureVector& filteredMissingFeatureVector = filteredDecorator->getView().secondView;

        for (uint32 i = numMinorityExamples; i < numExamples; i++) {
            if (i % 2 == 0) {
                EXPECT_TRUE(filteredMissingFeatureVector[i]);
            } else {
                EXPECT_FALSE(filteredMissingFeatureVector[i]);
            }
        }
    }
}

TEST(BinaryFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskUsingExisting) {
    uint32 numMinorityExamples = 10;
    AllocatedNominalFeatureVector featureVector(1, numMinorityExamples, 1);
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    AllocatedMissingFeatureVector missingFeatureVector;
    uint32 numMissingIndices = 10;
    uint32 numExamples = numMinorityExamples + numMissingIndices;

    for (uint32 i = numMinorityExamples; i < numExamples; i++) {
        missingFeatureVector.set(i, true);
    }

    CoverageMask coverageMask(numExamples);
    uint32 indicatorValue = 1;
    coverageMask.indicatorValue = indicatorValue;
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    for (uint32 i = 0; i < numExamples; i++) {
        if (i % 2 == 0) {
            coverageMaskIterator[i] = indicatorValue;
        }
    }

    std::unique_ptr<IFeatureVector> existing =
      std::make_unique<BinaryFeatureVectorDecorator>(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> filtered = existing->createFilteredFeatureVector(existing, coverageMask);
    const BinaryFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const BinaryFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);
    EXPECT_TRUE(existing.get() == nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const BinaryFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        BinaryFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(0);
        BinaryFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(0);
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
        const MissingFeatureVector& filteredMissingFeatureVector = filteredDecorator->getView().secondView;

        for (uint32 i = numMinorityExamples; i < numExamples; i++) {
            if (i % 2 == 0) {
                EXPECT_TRUE(filteredMissingFeatureVector[i]);
            } else {
                EXPECT_FALSE(filteredMissingFeatureVector[i]);
            }
        }
    }
}

TEST(BinaryFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskReturnsEqualFeatureVector) {
    uint32 numMinorityExamples = 10;
    AllocatedNominalFeatureVector featureVector(1, numMinorityExamples, 1);
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        indexIterator[i] = i;
    }

    CoverageMask coverageMask(numMinorityExamples);
    coverageMask.indicatorValue = 1;

    BinaryFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}
