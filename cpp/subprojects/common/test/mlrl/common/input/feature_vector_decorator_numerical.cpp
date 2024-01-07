#include "mlrl/common/input/feature_vector_decorator_numerical.hpp"

#include <gtest/gtest.h>

TEST(NumericalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromIndices) {
    uint32 numDenseExamples = 10;
    float32 sparseValue = -1;
    bool sparse = true;
    AllocatedNumericalFeatureVector featureVector(numDenseExamples, sparseValue, sparse);
    AllocatedNumericalFeatureVector::iterator iterator = featureVector.begin();

    for (uint32 i = 0; i < numDenseExamples; i++) {
        IndexedValue<float32>& entry = iterator[i];
        entry.index = i;
        entry.value = (float32) i;
    }

    NumericalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    uint32 start = 2;
    uint32 end = 7;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, start, end);
    const NumericalFeatureVectorView* filteredDecorator =
      dynamic_cast<const NumericalFeatureVectorView*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NumericalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        EXPECT_EQ(filteredFeatureVector.sparseValue, featureVector.sparseValue);
        EXPECT_EQ(filteredFeatureVector.sparse, featureVector.sparse);
        EXPECT_EQ(filteredFeatureVector.numElements, end - start);
        NumericalFeatureVector::const_iterator filteredIterator = filteredFeatureVector.cbegin();

        for (uint32 i = 0; i < end - start; i++) {
            const IndexedValue<float32>& entry = filteredIterator[i];
            EXPECT_EQ(entry.index, start + i);
            EXPECT_EQ(entry.value, (float32) start + i);
        }
    }
}

TEST(NumericalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromViewWithIndices) {
    uint32 numDenseExamples = 10;
    float32 sparseValue = -1;
    bool sparse = true;
    AllocatedNumericalFeatureVector featureVector(numDenseExamples, sparseValue, sparse);
    AllocatedNumericalFeatureVector::iterator iterator = featureVector.begin();

    for (uint32 i = 0; i < numDenseExamples; i++) {
        IndexedValue<float32>& entry = iterator[i];
        entry.index = i;
        entry.value = (float32) i;
    }

    NumericalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    uint32 start = 2;
    uint32 end = 7;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, 0, numDenseExamples)
                                                 ->createFilteredFeatureVector(existing, start, end);
    const NumericalFeatureVectorView* filteredDecorator =
      dynamic_cast<const NumericalFeatureVectorView*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NumericalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        EXPECT_EQ(filteredFeatureVector.sparseValue, featureVector.sparseValue);
        EXPECT_EQ(filteredFeatureVector.sparse, featureVector.sparse);
        EXPECT_EQ(filteredFeatureVector.numElements, end - start);
        NumericalFeatureVector::const_iterator filteredIterator = filteredFeatureVector.cbegin();

        for (uint32 i = 0; i < end - start; i++) {
            const IndexedValue<float32>& entry = filteredIterator[i];
            EXPECT_EQ(entry.index, start + i);
            EXPECT_EQ(entry.value, (float32) start + i);
        }
    }
}

TEST(NumericalFeatureVectorDecoratorTest, createFilteredFeatureVectorFromViewWithCoverageMask) {
    uint32 numDenseExamples = 10;
    float32 sparseValue = -1;
    bool sparse = true;
    AllocatedNumericalFeatureVector featureVector(numDenseExamples, sparseValue, sparse);
    AllocatedNumericalFeatureVector::iterator iterator = featureVector.begin();

    for (uint32 i = 0; i < numDenseExamples; i++) {
        IndexedValue<float32>& entry = iterator[i];
        entry.index = i;
        entry.value = (float32) i;
    }

    CoverageMask coverageMask(numDenseExamples);
    uint32 indicatorValue = 1;
    coverageMask.setIndicatorValue(indicatorValue);
    CoverageMask::iterator coverageMaskIterator = coverageMask.begin();

    for (uint32 i = 0; i < numDenseExamples; i++) {
        if (i % 2 == 0) {
            coverageMaskIterator[i] = indicatorValue;
        }
    }

    NumericalFeatureVectorDecorator decorator(std::move(featureVector), AllocatedMissingFeatureVector());
    std::unique_ptr<IFeatureVector> existing;
    std::unique_ptr<IFeatureVector> filtered = decorator.createFilteredFeatureVector(existing, 0, numDenseExamples)
                                                 ->createFilteredFeatureVector(existing, coverageMask);
    const NumericalFeatureVectorDecorator* filteredDecorator =
      dynamic_cast<const NumericalFeatureVectorDecorator*>(filtered.get());
    EXPECT_TRUE(filteredDecorator != nullptr);

    if (filteredDecorator) {
        // Check filtered indices...
        const NumericalFeatureVector& filteredFeatureVector = filteredDecorator->getView().firstView;
        EXPECT_EQ(filteredFeatureVector.sparseValue, featureVector.sparseValue);
        EXPECT_EQ(filteredFeatureVector.sparse, featureVector.sparse);
        EXPECT_EQ(filteredFeatureVector.numElements, numDenseExamples / 2);
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
    }
}

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

    if (filteredDecorator) {
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

    if (filteredDecorator) {
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
