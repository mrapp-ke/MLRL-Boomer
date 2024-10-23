#include "mlrl/common/data/vector_bit_binary.hpp"

#include <gtest/gtest.h>

TEST(BinaryBitVectorTest, getNumElements) {
    uint32 numElements = 270;
    BinaryBitVector vector(numElements);
    EXPECT_EQ(vector.getNumElements(), numElements);
}

TEST(BinaryBitVectorTest, defaultInitialization) {
    uint32 numElements = 270;
    BinaryBitVector vector(numElements, true);

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_FALSE(vector[i]);
    }
}

TEST(BinaryBitVectorTest, set) {
    uint32 numElements = 270;
    BinaryBitVector vector(numElements, false);

    for (uint32 i = 0; i < numElements; i++) {
        vector.set(i, false);
        EXPECT_FALSE(vector[i]);
    }
}

TEST(BinaryBitVectorTest, clear) {
    uint32 numElements = 270;
    BinaryBitVector vector(numElements);

    for (uint32 i = 0; i < numElements; i++) {
        vector.set(i, true);
        EXPECT_TRUE(vector[i]);
    }

    vector.clear();

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_FALSE(vector[i]);
    }
}
