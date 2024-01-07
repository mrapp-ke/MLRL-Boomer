#include "mlrl/common/input/feature_vector_decorator_nominal.hpp"

#include <gtest/gtest.h>

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
    uint32 start = 1;
    uint32 end = 3;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, start, end);
    const NominalFeatureVectorView* filteredDecorator = dynamic_cast<const NominalFeatureVectorView*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NominalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        EXPECT_EQ(filteredFeatureVector.numValues, end - start);
        NominalFeatureVector::value_const_iterator valuesBegin = filteredFeatureVector.values_cbegin();

        for (uint32 i = 0; i < end - start; i++) {
            EXPECT_EQ(valuesBegin[i], start + i);
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue);

            for (uint32 j = 0; j < numIndices; j++) {
                EXPECT_EQ(indicesBegin[j], ((i + start) * numExamplesPerValue) + j);
            }
        }
    }
}

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromViewWithIndices) {
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
    uint32 start = 1;
    uint32 end = 3;
    std::unique_ptr<IFeatureVector> filtered =
      decorator.createFilteredFeatureVector(existing, 0, numValues)->createFilteredFeatureVector(existing, start, end);
    const NominalFeatureVectorView* filteredDecorator = dynamic_cast<const NominalFeatureVectorView*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NominalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        EXPECT_EQ(filteredFeatureVector.numValues, end - start);
        NominalFeatureVector::value_const_iterator valuesBegin = filteredFeatureVector.values_cbegin();

        for (uint32 i = 0; i < end - start; i++) {
            EXPECT_EQ(valuesBegin[i], start + i);
            NominalFeatureVector::index_const_iterator indicesBegin = filteredFeatureVector.indices_cbegin(i);
            NominalFeatureVector::index_const_iterator indicesEnd = filteredFeatureVector.indices_cend(i);
            uint32 numIndices = indicesEnd - indicesBegin;
            EXPECT_EQ(numIndices, numExamplesPerValue);

            for (uint32 j = 0; j < numIndices; j++) {
                EXPECT_EQ(indicesBegin[j], ((i + start) * numExamplesPerValue) + j);
            }
        }
    }
}

TEST(NominalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromViewWithCoverageMask) {
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

    CoverageMask coverageMask(numMinorityExamples);
    uint32 indicatorValue = 1;
    coverageMask.setIndicatorValue(indicatorValue);
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    for (uint32 i = 0; i < numMinorityExamples; i++) {
        if (i % 2 == 0) {
            coverageMaskIterator[i] = indicatorValue;
        }
    }

    NominalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, 0, numValues)
                                                 ->createFilteredFeatureVector(existing, coverageMask);
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
    coverageMask.setIndicatorValue(indicatorValue);
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
    coverageMask.setIndicatorValue(indicatorValue);
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
    coverageMask.setIndicatorValue(1);

    NominalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}
