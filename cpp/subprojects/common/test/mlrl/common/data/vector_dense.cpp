#include "mlrl/common/data/vector_dense.hpp"

#include <gtest/gtest.h>

TEST(DenseVectorTest, getNumElements) {
    uint32 numElements = 10;
    DenseVector<uint32> vector(numElements);
    EXPECT_EQ(vector.getNumElements(), numElements);
}

TEST(DenseVectorTest, defaultInitialization) {
    uint32 numElements = 10;
    DenseVector<uint32> vector(numElements, true);

    for (uint32 i = 0; i < numElements; i++) {
        EXPECT_EQ(vector[i], (uint32) 0);
    }
}

TEST(DenseVectorTest, writeAccess) {
    uint32 numElements = 10;
    DenseVector<uint32> vector(numElements);

    for (uint32 i = 0; i < numElements; i++) {
        vector[i] = 0;
        EXPECT_EQ(vector[i], (uint32) 0);
    }
}

TEST(DenseVectorTest, iteratorAccess) {
    DenseVector<uint32> vector(10);

    for (DenseVector<uint32>::iterator it = vector.begin(); it != vector.end(); it++) {
        *it = 0;
    }

    for (DenseVector<uint32>::const_iterator it = vector.cbegin(); it != vector.cend(); it++) {
        EXPECT_EQ(*it, (uint32) 0);
    }
}

TEST(ResizableDenseVectorTest, setNumElements) {
    ResizableDenseVector<uint32> vector(10);
    uint32 numElements = 15;
    vector.setNumElements(numElements, true);
    EXPECT_EQ(vector.getNumElements(), numElements);
}
