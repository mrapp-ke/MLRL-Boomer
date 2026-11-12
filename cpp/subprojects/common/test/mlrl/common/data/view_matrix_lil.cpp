#include "mlrl/common/data/view_matrix_lil.hpp"

#include <gtest/gtest.h>

TEST(AllocatedListOfListsTest, numRows) {
    uint32 numRows = 10;
    AllocatedListOfLists<uint32> lilMatrix(numRows, 2);
    EXPECT_EQ(lilMatrix.numRows, numRows);
}

TEST(AllocatedListOfListsTest, numCols) {
    uint32 numCols = 10;
    AllocatedListOfLists<uint32> lilMatrix(2, numCols);
    EXPECT_EQ(lilMatrix.numCols, numCols);
}

TEST(AllocatedListOfListsTest, emplace_back) {
    uint32 numRows = 10;
    uint32 numCols = 10;
    AllocatedListOfLists<uint32> lilMatrix(numRows, numCols);

    for (uint32 i = 0; i < numRows; i++) {
        auto row = lilMatrix[i];

        for (uint32 j = 0; j < numCols; j++) {
            row.emplace_back(j);
        }
    }

    for (uint32 i = 0; i < numRows; i++) {
        auto rowIterator = lilMatrix.values_cbegin(i);
        EXPECT_EQ(numCols, lilMatrix.values_cend(i) - rowIterator);
        EXPECT_EQ(numCols, lilMatrix[i].numElements);

        for (uint32 j = 0; j < numCols; j++) {
            EXPECT_EQ(rowIterator[j], j);
        }
    }
}

TEST(AllocatedListOfListsTest, emplace) {
    uint32 numRows = 10;
    uint32 numCols = 10;
    AllocatedListOfLists<uint32> lilMatrix(numRows, numCols);

    for (uint32 i = 0; i < numRows; i++) {
        auto row = lilMatrix[i];

        for (uint32 j = 0; j < numCols; j++) {
            row.emplace(row.begin(), j);
        }
    }

    for (uint32 i = 0; i < numRows; i++) {
        auto rowIterator = lilMatrix.values_cbegin(i);
        EXPECT_EQ(numCols, lilMatrix.values_cend(i) - rowIterator);
        EXPECT_EQ(numCols, lilMatrix[i].numElements);

        for (uint32 j = 0; j < numCols; j++) {
            EXPECT_EQ(rowIterator[j], numCols - j - 1);
        }
    }
}

TEST(AllocatedListOfListsTest, erase) {
    uint32 numRows = 10;
    uint32 numCols = 10;
    AllocatedListOfLists<uint32> lilMatrix(numRows, numCols);

    for (uint32 i = 0; i < numRows; i++) {
        auto row = lilMatrix[i];

        for (uint32 j = 0; j < numCols; j++) {
            row.emplace_back(j);
        }
    }

    for (uint32 i = 0; i < numRows; i++) {
        auto row = lilMatrix[i];
        EXPECT_EQ(numCols, row.numElements);

        for (uint32 j = 0; j < numCols; j++) {
            auto it = row.erase(row.begin());

            if (it != row.end()) {
                EXPECT_EQ(*it, j + 1);
            }
        }

        EXPECT_EQ(0, row.cend() - row.cbegin());
        EXPECT_EQ((uint32) 0, row.numElements);
    }
}

TEST(AllocatedListOfListsTest, clear) {
    uint32 numRows = 10;
    uint32 numCols = 10;
    AllocatedListOfLists<uint32> lilMatrix(numRows, numCols);

    for (uint32 i = 0; i < numRows; i++) {
        auto row = lilMatrix[i];

        for (uint32 j = 0; j < numCols; j++) {
            row.emplace_back(j);
        }
    }

    for (uint32 i = 0; i < numRows; i++) {
        auto row = lilMatrix[i];
        row.clear();
        EXPECT_EQ(0, row.cend() - row.cbegin());
        EXPECT_EQ((uint32) 0, row.numElements);
        EXPECT_EQ(0, lilMatrix.values_cend(i) - lilMatrix.values_cbegin(i));
    }
}
