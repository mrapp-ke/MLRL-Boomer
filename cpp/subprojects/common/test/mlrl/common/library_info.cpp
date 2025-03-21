#include "mlrl/common/library_info.hpp"

#include <gtest/gtest.h>

TEST(InfoTest, getLibraryInfo) {
    std::unique_ptr<ILibraryInfo> libraryInfoPtr = getLibraryInfo();
    EXPECT_EQ(libraryInfoPtr->getLibraryName(), "libmlrlcommon");
    EXPECT_NE(libraryInfoPtr->getLibraryVersion(), "");
    EXPECT_NE(libraryInfoPtr->getTargetArchitecture(), "");
}
