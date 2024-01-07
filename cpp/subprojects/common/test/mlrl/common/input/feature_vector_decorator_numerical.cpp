#include "mlrl/common/input/feature_vector_decorator_numerical.hpp"

#include <gtest/gtest.h>

TEST(NumericalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMask) {
    uint32 numDenseExamples = 10;
    AllocatedNumericalFeatureVector featureVector(numDenseExamples);
    AllocatedNumericalFeatureVector::iterator iterator = featureVector.begin();

    for (uint32 i = 0; i < numDenseExamples; i++) {
        IndexedValue<float32>& entry = iterator[i];
        entry.index = i;
        entry.value = (float32) i;
    }

    AllocatedMissingFeatureVector missingFeatureVector;
    uint32 numMissingIndices = 10;
    uint32 numExamples = numDenseExamples + numMissingIndices;

    for (uint32 i = numDenseExamples; i < numExamples; i++) {
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

    NumericalFeatureVectorDecorator decorator(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const NumericalFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const NumericalFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    // Check filtered indices...
    const NumericalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
    std::unordered_set<uint32> indices;

    for (auto it = filteredFeatureVector.cbegin(); it != filteredFeatureVector.cend(); it++) {
        const IndexedValue<float32>& entry = *it;
        indices.emplace(entry.index);
    }

    for (uint32 i = 0; i < numDenseExamples; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(indices.find(i) != indices.end());
        } else {
            EXPECT_TRUE(indices.find(i) == indices.end());
        }
    }

    // Check missing indices...
    const MissingFeatureVector& filteredMissingFeatureVector = filteredDecorator->getView().secondView;

    for (uint32 i = numDenseExamples; i < numExamples; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(filteredMissingFeatureVector[i]);
        } else {
            EXPECT_FALSE(filteredMissingFeatureVector[i]);
        }
    }
}

TEST(NumericalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskUsingExisting) {
    uint32 numDenseExamples = 10;
    AllocatedNumericalFeatureVector featureVector(numDenseExamples);
    AllocatedNumericalFeatureVector::iterator iterator = featureVector.begin();

    for (uint32 i = 0; i < numDenseExamples; i++) {
        IndexedValue<float32>& entry = iterator[i];
        entry.index = i;
        entry.value = (float32) i;
    }

    AllocatedMissingFeatureVector missingFeatureVector;
    uint32 numMissingIndices = 10;
    uint32 numExamples = numDenseExamples + numMissingIndices;

    for (uint32 i = numDenseExamples; i < numExamples; i++) {
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
      std::make_unique<NumericalFeatureVectorDecorator>(std::move(featureVector), std::move(missingFeatureVector));
    std::unique_ptr<IFeatureVector> filtered = existing->createFilteredFeatureVector(existing, coverageMask);
    const NumericalFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const NumericalFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);
    EXPECT_TRUE(existing.get() == nullptr);

    // Check filtered indices...
    const NumericalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;

    std::unordered_set<uint32> indices;

    for (auto it = filteredFeatureVector.cbegin(); it != filteredFeatureVector.cend(); it++) {
        const IndexedValue<float32>& entry = *it;
        indices.emplace(entry.index);
    }

    for (uint32 i = 0; i < numDenseExamples; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(indices.find(i) != indices.end());
        } else {
            EXPECT_TRUE(indices.find(i) == indices.end());
        }
    }

    // Check missing indices...
    const MissingFeatureVector& filteredMissingFeatureVector = filteredDecorator->getView().secondView;

    for (uint32 i = numDenseExamples; i < numExamples; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(filteredMissingFeatureVector[i]);
        } else {
            EXPECT_FALSE(filteredMissingFeatureVector[i]);
        }
    }
}

TEST(NumericalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromCoverageMaskReturnsEqualFeatureVector) {
    uint32 numDenseExamples = 10;
    AllocatedNumericalFeatureVector featureVector(numDenseExamples);
    AllocatedNumericalFeatureVector::iterator iterator = featureVector.begin();

    for (uint32 i = 0; i < numDenseExamples; i++) {
        IndexedValue<float32>& entry = iterator[i];
        entry.index = i;
        entry.value = (float32) i;
    }

    CoverageMask coverageMask(numDenseExamples);
    coverageMask.setIndicatorValue(1);

    NumericalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, coverageMask);
    const EqualFeatureVector* filteredFeatureVector = dynamic_cast<const EqualFeatureVector*>(filtered.get());
    EXPECT_TRUE(filteredFeatureVector != nullptr);
}
