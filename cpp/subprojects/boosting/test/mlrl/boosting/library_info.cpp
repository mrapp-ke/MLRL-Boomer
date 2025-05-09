#include "mlrl/boosting/library_info.hpp"

#include <gtest/gtest.h>

namespace boosting {

    TEST(InfoTest, getLibraryInfo) {
        std::unique_ptr<ILibraryInfo> libraryInfoPtr = getLibraryInfo();
        EXPECT_EQ(libraryInfoPtr->getLibraryName(), "libmlrlboosting");
        EXPECT_NE(libraryInfoPtr->getLibraryVersion(), "");
        EXPECT_NE(libraryInfoPtr->getTargetArchitecture(), "");
    }

}
