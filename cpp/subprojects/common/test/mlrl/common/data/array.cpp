#include "mlrl/common/data/array.hpp"

#include <gtest/gtest.h>

TEST(ArrayTest, defaultInitialization) {
    uint32 numElements = 10;
    Array<uint32> array(numElements, true);

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_EQ(array[i], (uint32) 0);
    }
}

TEST(ArrayTest, writeAccess) {
    uint32 numElements = 10;
    Array<uint32> array(numElements);

    for (uint32 i = 0; i < numElements; i++) {
        array[i] = 0;
        EXPECT_EQ(array[i], (uint32) 0);
    }
}
