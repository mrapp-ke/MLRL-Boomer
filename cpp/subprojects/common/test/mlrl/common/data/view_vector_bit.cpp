#include "mlrl/common/data/view_vector_bit.hpp"

#include <gtest/gtest.h>

TEST(BitVectorTest, constructor) {
    uint32 numElements = 256;
    uint32 numBitsPerElement = 4;
    AllocatedBitVector<uint32> vector(numElements, numBitsPerElement);
    EXPECT_EQ(vector.numElements, numElements);
    EXPECT_EQ(vector.numBitsPerElement, numBitsPerElement);
}

TEST(BitVectorTest, defaultInitialization) {
    uint32 numElements = 256;
    AllocatedBitVector<uint32> vector(numElements, 4, true);

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_EQ(vector[i], (uint32) 0);
    }
}

TEST(BitVectorTest, set1) {
    uint32 numElements = 256;
    uint32 numBitsPerElement = 4;
    uint32 numValuesPerElement = util::getNumBitCombinations(numBitsPerElement);
    AllocatedBitVector<uint32> vector(numElements, numBitsPerElement, false);
    uint32 i = 0;

    while (i < numElements) {
        for (uint32 j = 0; i < numElements && j < numValuesPerElement; j++) {
            vector.set(i, j);
            i++;
        }
    }

    i = 0;

    while (i < numElements) {
        for (uint32 j = 0; i < numElements && j < numValuesPerElement; j++) {
            EXPECT_EQ(vector[i], (uint32) j);
            i++;
        }
    }
}

TEST(BitVectorTest, set2) {
    uint32 numElements = 256;
    uint32 numBitsPerElement = 3;
    uint32 numValuesPerElement = util::getNumBitCombinations(numBitsPerElement);
    AllocatedBitVector<uint32> vector(numElements, numBitsPerElement, false);
    uint32 i = 0;

    while (i < numElements) {
        for (uint32 j = 0; i < numElements && j < numValuesPerElement; j++) {
            vector.set(i, j);
            i++;
        }
    }

    i = 0;

    while (i < numElements) {
        for (uint32 j = 0; i < numElements && j < numValuesPerElement; j++) {
            EXPECT_EQ(vector[i], (uint32) j);
            i++;
        }
    }
}

TEST(BitVectorTest, clear) {
    uint32 numElements = 256;
    AllocatedBitVector<uint32> vector(numElements, 4, false);
    vector.clear();

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_EQ(vector[i], (uint32) 0);
    }
}
