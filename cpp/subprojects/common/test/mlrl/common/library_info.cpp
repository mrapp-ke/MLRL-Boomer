#include "mlrl/common/library_info.hpp"

#include <gtest/gtest.h>

TEST(InfoTest, getLibraryInfo) {
    std::unique_ptr<ILibraryInfo> libraryInfoPtr = getLibraryInfo();
    EXPECT_EQ(libraryInfoPtr->getLibraryName(), "libmlrlcommon");
    EXPECT_NE(libraryInfoPtr->getLibraryVersion(), "");
    EXPECT_NE(libraryInfoPtr->getTargetArchitecture(), "");
}

TEST(InfoTest, getHdf5Version) {
    std::unique_ptr<ILibraryInfo> libraryInfoPtr = getLibraryInfo();
    std::string hdf5Version;
    libraryInfoPtr->visitBuildOptions([&](const ILibraryInfo::BuildOption& buildOption) {
        if (buildOption.option == "HDF5_VERSION") {
            hdf5Version = buildOption.value;
        }
    });
    EXPECT_NE(hdf5Version, "");
}
