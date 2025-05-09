#include "mlrl/common/input/feature_vector_decorator_nominal.hpp"

#include "statistics_weighted.hpp"

#include <gtest/gtest.h>

TEST(NominalFeatureVectorDecoratorTest, updateCoverageMaskAndStatistics) {
    uint32 numValues = 4;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(numValues, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::value_iterator valueIterator = featureVector.values;
    AllocatedNominalFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        valueIterator[i] = i;
        indptrIterator[i] = i * numExamplesPerValue;

        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
            indexIterator[index] = index;
        }
    }

    WeightedStatistics statistics;
    uint32 numExamples = numMinorityExamples + 15;

    for (uint32 i = 0; i < numExamples; i++) {
        statistics.addCoveredStatistic(i);
    }

    NominalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    Interval interval(1, 3);
    CoverageMask coverageMask(numExamples);
    uint32 indicatorValue = 1;
    decorator.updateCoverageMaskAndStatistics(interval, coverageMask, indicatorValue, statistics);
    EXPECT_EQ(coverageMask.indicatorValue, indicatorValue);
    const NominalFeatureVector& nominalFeatureVector = decorator.getView().firstView;

    for (uint32 i = 0; i < interval.start; i++) {
        for (auto it = nominalFeatureVector.indices_cbegin(i); it != nominalFeatureVector.indices_cend(i); it++) {
            uint32 index = *it;
            EXPECT_FALSE(coverageMask[index]);
            EXPECT_FALSE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
        }
    }

    for (uint32 i = interval.start; i < interval.end; i++) {
        for (auto it = nominalFeatureVector.indices_cbegin(i); it != nominalFeatureVector.indices_cend(i); it++) {
            uint32 index = *it;
            EXPECT_TRUE(coverageMask[index]);
            EXPECT_TRUE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
        }
    }

    for (uint32 i = interval.end; i < numValues; i++) {
        for (auto it = nominalFeatureVector.indices_cbegin(i); it != nominalFeatureVector.indices_cend(i); it++) {
            uint32 index = *it;
            EXPECT_FALSE(coverageMask[index]);
            EXPECT_FALSE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
        }
    }

    for (uint32 i = numMinorityExamples; i < numExamples; i++) {
        EXPECT_FALSE(coverageMask[i]);
        EXPECT_FALSE(statistics.coveredStatistics.find(i) != statistics.coveredStatistics.end());
    }
}

TEST(NominalFeatureVectorDecoratorTest, updateCoverageMaskAndStatisticsInverse) {
    uint32 numValues = 4;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(numValues, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::value_iterator valueIterator = featureVector.values;
    AllocatedNominalFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        valueIterator[i] = i;
        indptrIterator[i] = i * numExamplesPerValue;

        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
            indexIterator[index] = index;
        }
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

    NominalFeatureVectorDecorator decorator(std::move(featureVector), std::move(missingFeatureVector));
    Interval interval(1, 3, true);
    CoverageMask coverageMask(numExamples);
    uint32 indicatorValue = 1;
    decorator.updateCoverageMaskAndStatistics(interval, coverageMask, indicatorValue, statistics);
    EXPECT_EQ(coverageMask.indicatorValue, (uint32) 0);
    const NominalFeatureVector& nominalFeatureVector = decorator.getView().firstView;

    for (uint32 i = 0; i < interval.start; i++) {
        for (auto it = nominalFeatureVector.indices_cbegin(i); it != nominalFeatureVector.indices_cend(i); it++) {
            uint32 index = *it;
            EXPECT_TRUE(coverageMask[index]);
            EXPECT_TRUE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
        }
    }

    for (uint32 i = interval.start; i < interval.end; i++) {
        for (auto it = nominalFeatureVector.indices_cbegin(i); it != nominalFeatureVector.indices_cend(i); it++) {
            uint32 index = *it;
            EXPECT_FALSE(coverageMask[index]);
            EXPECT_FALSE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
        }
    }

    for (uint32 i = interval.end; i < numValues; i++) {
        for (auto it = nominalFeatureVector.indices_cbegin(i); it != nominalFeatureVector.indices_cend(i); it++) {
            uint32 index = *it;
            EXPECT_TRUE(coverageMask[index]);
            EXPECT_TRUE(statistics.coveredStatistics.find(index) != statistics.coveredStatistics.end());
        }
    }

    for (uint32 i = numMinorityExamples; i < numMinorityExamples + numMissingExamples; i++) {
        EXPECT_FALSE(coverageMask[i]);
        EXPECT_FALSE(statistics.coveredStatistics.find(i) != statistics.coveredStatistics.end());
    }

    for (uint32 i = numMinorityExamples + numMissingExamples; i < numExamples; i++) {
        EXPECT_TRUE(coverageMask[i]);
        EXPECT_TRUE(statistics.coveredStatistics.find(i) != statistics.coveredStatistics.end());
    }
}

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromIndices) {
    uint32 numValues = 4;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(numValues, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::value_iterator valueIterator = featureVector.values;
    AllocatedNominalFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        valueIterator[i] = i;
        indptrIterator[i] = i * numExamplesPerValue;

        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
            indexIterator[index] = index;
        }
    }

    NominalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, Interval(1, 2));
    const EqualFeatureVector* filteredDecorator = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);
}

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromIndicesInverse) {
    uint32 numValues = 4;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(numValues, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::value_iterator valueIterator = featureVector.values;
    AllocatedNominalFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        valueIterator[i] = i;
        indptrIterator[i] = i * numExamplesPerValue;

        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
            indexIterator[index] = index;
        }
    }

    NominalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    Interval interval(1, 2, true);
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, interval);
    const NominalFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const NominalFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NominalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        EXPECT_EQ(filteredFeatureVector.numBins, interval.start + (numValues - interval.end));
        NominalFeatureVector::value_const_iterator valuesBegin = filteredFeatureVector.values_cbegin();
        uint32 n = 0;

        for (uint32 i = 0; i < interval.start; i++) {
            EXPECT_EQ(valuesBegin[n], (int32) i);
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(n);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(n);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue);

            for (uint32 j = 0; j < numIndices; j++) {
                EXPECT_EQ(indicesBegin[j], (i * numExamplesPerValue) + j);
            }

            n++;
        }

        for (uint32 i = interval.end; i < featureVector.numBins; i++) {
            EXPECT_EQ(valuesBegin[n], (int32) i);
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(n);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(n);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue);

            for (uint32 j = 0; j < numIndices; j++) {
                EXPECT_EQ(indicesBegin[j], (i * numExamplesPerValue) + j);
            }

            n++;
        }
    }
}

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromIndicesInverseUsingExisting) {
    uint32 numValues = 4;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(numValues, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::value_iterator valueIterator = featureVector.values;
    AllocatedNominalFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        valueIterator[i] = i;
        indptrIterator[i] = i * numExamplesPerValue;

        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
            indexIterator[index] = index;
        }
    }

    std::unique_ptr<IFeatureVector> existing =
      std::make_unique<NominalFeatureVectorDecorator>(std::move(featureVector), AllocatedMissingFeatureVector());
    Interval interval(1, 2, true);
    std::unique_ptr<IFeatureVector> filtered = existing->createFilteredFeatureVector(existing, interval);
    const NominalFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const NominalFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NominalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        EXPECT_EQ(filteredFeatureVector.numBins, interval.start + (numValues - interval.end));
        NominalFeatureVector::value_const_iterator valuesBegin = filteredFeatureVector.values_cbegin();
        uint32 n = 0;

        for (uint32 i = 0; i < interval.start; i++) {
            EXPECT_EQ(valuesBegin[n], (int32) i);
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(n);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(n);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue);

            for (uint32 j = 0; j < numIndices; j++) {
                EXPECT_EQ(indicesBegin[j], (i * numExamplesPerValue) + j);
            }

            n++;
        }

        for (uint32 i = interval.end; i < featureVector.numBins; i++) {
            EXPECT_EQ(valuesBegin[n], (int32) i);
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(n);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(n);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue);

            for (uint32 j = 0; j < numIndices; j++) {
                EXPECT_EQ(indicesBegin[j], (i * numExamplesPerValue) + j);
            }

            n++;
        }
    }
}

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMask) {
    uint32 numValues = 3;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(numValues, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::value_iterator valueIterator = featureVector.values;
    AllocatedNominalFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        valueIterator[i] = i;
        indptrIterator[i] = i * numExamplesPerValue;

        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
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

    NominalFeatureVectorDecorator decorator(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const NominalFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const NominalFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NominalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;

        for (uint32 i = 0; i < numValues; i++) {
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue / 2);

            std::unordered_set<uint32> indices;

            for (auto it = indicesBegin; it != indicesEnd; it++) {
                indices.emplace(*it);
            }

            for (uint32 j = 0; j < numExamplesPerValue; j++) {
                uint32 index = (i * numExamplesPerValue) + j;

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

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskUsingExisting) {
    uint32 numValues = 3;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(numValues, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::value_iterator valueIterator = featureVector.values;
    AllocatedNominalFeatureVector::index_iterator indptrIterator = featureVector.indptr;
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        valueIterator[i] = i;
        indptrIterator[i] = i * numExamplesPerValue;

        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
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
      std::make_unique<NominalFeatureVectorDecorator>(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> filtered = existing->createFilteredFeatureVector(existing, coverageMask);
    const NominalFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const NominalFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);
    EXPECT_TRUE(existing.get() == nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NominalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;

        for (uint32 i = 0; i < numValues; i++) {
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue / 2);

            std::unordered_set<uint32> indices;

            for (auto it = indicesBegin; it != indicesEnd; it++) {
                indices.emplace(*it);
            }

            for (uint32 j = 0; j < numExamplesPerValue; j++) {
                uint32 index = (i * numExamplesPerValue) + j;

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

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskReturnsEqualFeatureVector) {
    uint32 numValues = 1;
    uint32 numExamplesPerValue = 10;
    uint32 numMinorityExamples = numValues * numExamplesPerValue;
    AllocatedNominalFeatureVector featureVector(1, numMinorityExamples, 0);
    AllocatedNominalFeatureVector::index_iterator indexIterator = featureVector.indices_begin(0);

    for (uint32 i = 0; i < numValues; i++) {
        for (uint32 j = 0; j < numExamplesPerValue; j++) {
            uint32 index = (i * numExamplesPerValue) + j;
            indexIterator[index] = index;
        }
    }

    CoverageMask coverageMask(numMinorityExamples);
    coverageMask.indicatorValue = 1;

    NominalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}
