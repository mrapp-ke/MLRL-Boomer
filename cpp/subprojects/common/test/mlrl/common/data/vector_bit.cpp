#include "mlrl/common/data/vector_bit.hpp"

#include <gtest/gtest.h>

TEST(BitVectorTest, getNumElements) {
    uint32 numElements = 270;
    BitVector vector(numElements);
    EXPECT_EQ(vector.getNumElements(), numElements);
}

TEST(BitVectorTest, defaultInitialization) {
    uint32 numElements = 270;
    BitVector vector(numElements, true);

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_FALSE(vector[i]);
    }
}

TEST(BitVectorTest, set) {
    uint32 numElements = 270;
    BitVector vector(numElements, false);

    for (uint32 i = 0; i < numElements; i++) {
        vector.set(i, false);
        EXPECT_FALSE(vector[i]);
    }
}

TEST(BitVectorTest, clear) {
    uint32 numElements = 270;
    BitVector vector(numElements);

    for (uint32 i = 0; i < numElements; i++) {
        vector.set(i, true);
        EXPECT_TRUE(vector[i]);
    }

    vector.clear();

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_FALSE(vector[i]);
    }
}
