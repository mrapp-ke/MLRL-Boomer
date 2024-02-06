#include "mlrl/common/input/feature_vector_decorator_binned.hpp"

#include "mlrl/common/input/feature_vector_binned.hpp"
#include "statistics_weighted.hpp"

#include <gtest/gtest.h>

TEST(BinnedFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMask) {
    uint32 numBins = 3;
    uint32 numExamplesPerBin = 10;
    uint32 numMinorityExamples = numBins * numExamplesPerBin;
    AllocatedBinnedFeatureVector featureVector(numBins, numMinorityExamples, 0);
    AllocatedBinnedFeatureVector::threshold_iterator thresholdIterator = featureVector.thresholds;
    AllocatedBinnedFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedBinnedFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numBins; i++) {
        if (i < numBins - 1) {
            thresholdIterator[i] = i;
        }

        indptrIterator[i] = i * numExamplesPerBin;

        for (uint32 j = 0; j < numExamplesPerBin; j++) {
            uint32 index = (i * numExamplesPerBin) + j;
            indexIterator[index] = index;
        }
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

    BinnedFeatureVectorDecorator decorator(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const BinnedFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const BinnedFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const BinnedFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;

        for (uint32 i = 0; i < numBins; i++) {
            BinnedFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(i);
            BinnedFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerBin / 2);

            std::unordered_set<uint32> indices;

            for (auto it = indicesBegin; it != indicesEnd; it++) {
                indices.emplace(*it);
            }

            for (uint32 j = 0; j < numExamplesPerBin; j++) {
                uint32 index = (i * numExamplesPerBin) + j;

                if (index % 2 == 0) {
                    EXPECT_TRUE(indices.find(index) != indices.end());
                } else {
                    EXPECT_TRUE(indices.find(index) == indices.end());
                }
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

TEST(BinnedFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskUsingExisting) {
    uint32 numBins = 3;
    uint32 numExamplesPerBin = 10;
    uint32 numMinorityExamples = numBins * numExamplesPerBin;
    AllocatedBinnedFeatureVector featureVector(numBins, numMinorityExamples, 0);
    AllocatedBinnedFeatureVector::threshold_iterator thresholdIterator = featureVector.thresholds;
    AllocatedBinnedFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedBinnedFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numBins; i++) {
        if (i < numBins - 1) {
            thresholdIterator[i] = i;
        }

        indptrIterator[i] = i * numExamplesPerBin;

        for (uint32 j = 0; j < numExamplesPerBin; j++) {
            uint32 index = (i * numExamplesPerBin) + j;
            indexIterator[index] = index;
        }
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
      std::make_unique<BinnedFeatureVectorDecorator>(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> filtered = existing->createFilteredFeatureVector(existing, coverageMask);
    const BinnedFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const BinnedFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);
    EXPECT_TRUE(existing.get() == nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const BinnedFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;

        for (uint32 i = 0; i < numBins; i++) {
            BinnedFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(i);
            BinnedFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerBin / 2);

            std::unordered_set<uint32> indices;

            for (auto it = indicesBegin; it != indicesEnd; it++) {
                indices.emplace(*it);
            }

            for (uint32 j = 0; j < numExamplesPerBin; j++) {
                uint32 index = (i * numExamplesPerBin) + j;

                if (index % 2 == 0) {
                    EXPECT_TRUE(indices.find(index) != indices.end());
                } else {
                    EXPECT_TRUE(indices.find(index) == indices.end());
                }
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

TEST(BinnedFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskReturnsEqualFeatureVector) {
    uint32 numBins = 1;
    uint32 numExamplesPerBin = 10;
    uint32 numMinorityExamples = numBins * numExamplesPerBin;
    AllocatedBinnedFeatureVector featureVector(1, numMinorityExamples, 0);
    AllocatedBinnedFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numBins; i++) {
        for (uint32 j = 0; j < numExamplesPerBin; j++) {
            uint32 index = (i * numExamplesPerBin) + j;
            indexIterator[index] = index;
        }
    }

    CoverageMask coverageMask(numMinorityExamples);
    coverageMask.indicatorValue = 1;

    BinnedFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}
