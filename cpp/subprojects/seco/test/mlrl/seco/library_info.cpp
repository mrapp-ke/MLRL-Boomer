#include "mlrl/seco/library_info.hpp"

#include <gtest/gtest.h>

namespace seco {

    TEST(InfoTest, getLibraryInfo) {
        std::unique_ptr<ILibraryInfo> libraryInfoPtr = getLibraryInfo();
        EXPECT_EQ(libraryInfoPtr->getLibraryName(), "libmlrlseco");
        EXPECT_NE(libraryInfoPtr->getLibraryVersion(), "");
        EXPECT_NE(libraryInfoPtr->getTargetArchitecture(), "");
    }

}
