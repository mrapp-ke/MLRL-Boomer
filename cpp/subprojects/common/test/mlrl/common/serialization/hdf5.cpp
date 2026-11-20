#include "mlrl/common/serialization/hdf5/hdf5_file.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

template<typename T>
static void writeAndReadDataset(const std::vector<T>& values) {
    Hdf5File file = Hdf5File::createInMemory();
    file.getRootGroup().writeDataset<T>("dataset", values.data(), values.size());
    std::vector<uint8> image = file.getImage();
    Hdf5File loadedFile = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group root = loadedFile.getRootGroup();
    EXPECT_EQ(root.getDatasetSize("dataset"), values.size());
    EXPECT_EQ(root.readDataset<T>("dataset"), values);
}

static std::vector<uint8> createImageWithContent() {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group root = file.getRootGroup();
    std::vector<uint32> values = {1, 2, 3};
    root.writeDataset<uint32>("uint32", values.data(), values.size());
    std::vector<std::string> strings = {"a", "b"};
    root.writeStringDataset("strings", strings);
    root.setIntAttribute("int", 42);
    root.setFloatAttribute("float", 0.5);
    root.setStringAttribute("string", "value");
    root.createGroup("group");
    return file.getImage();
}

TEST(Hdf5Test, writeAndReadNumericDatasets) {
    writeAndReadDataset<uint8>({0, 1, 255});
    writeAndReadDataset<uint32>({0, 1, std::numeric_limits<uint32>::max()});
    writeAndReadDataset<int32>({std::numeric_limits<int32>::min(), -1, 0, std::numeric_limits<int32>::max()});
    writeAndReadDataset<std::int64_t>(
      {std::numeric_limits<std::int64_t>::min(), -1, 0, std::numeric_limits<std::int64_t>::max()});
    writeAndReadDataset<float32>({-1.5f, 0.0f, 3.25f, std::numeric_limits<float32>::infinity()});
    writeAndReadDataset<float64>({-1.5, 0.0, 0.1, std::numeric_limits<float64>::max()});
}

TEST(Hdf5Test, writeAndReadEmptyDatasets) {
    writeAndReadDataset<uint8>({});
    writeAndReadDataset<uint32>({});
    writeAndReadDataset<float64>({});

    Hdf5File file = Hdf5File::createInMemory();
    file.getRootGroup().writeDataset<float32>("null", nullptr, 0);
    file.getRootGroup().writeStringDataset("strings", {});
    std::vector<uint8> image = file.getImage();
    Hdf5File loadedFile = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group root = loadedFile.getRootGroup();
    EXPECT_TRUE(root.readDataset<float32>("null").empty());
    EXPECT_TRUE(root.readStringDataset("strings").empty());
}

TEST(Hdf5Test, writeAndReadStringDatasets) {
    std::vector<std::string> values = {"", "ascii", "\xC3\xA4\xC3\xB6\xC3\xBC \xE2\x82\xAC \xF0\x9F\x98\x80",
                                       std::string(1000, 'x')};
    Hdf5File file = Hdf5File::createInMemory();
    file.getRootGroup().writeStringDataset("strings", values);
    std::vector<uint8> image = file.getImage();
    Hdf5File loadedFile = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group root = loadedFile.getRootGroup();
    EXPECT_EQ(root.getDatasetTypeClass("strings"), Hdf5TypeClass::STRING);
    EXPECT_EQ(root.readStringDataset("strings"), values);
}

TEST(Hdf5Test, writeAndReadAttributes) {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group root = file.getRootGroup();
    root.setIntAttribute("int_min", std::numeric_limits<std::int64_t>::min());
    root.setIntAttribute("int_max", std::numeric_limits<std::int64_t>::max());
    root.setFloatAttribute("float", -0.1);
    root.setStringAttribute("string", "\xC3\xA4\xC3\xB6\xC3\xBC");
    root.setStringAttribute("empty_string", "");
    root.setIntAttribute("replaced", 1);
    root.setStringAttribute("replaced", "replacement");
    std::vector<uint8> image = file.getImage();

    Hdf5File loadedFile = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group loadedRoot = loadedFile.getRootGroup();
    EXPECT_EQ(loadedRoot.getIntAttribute("int_min"), std::numeric_limits<std::int64_t>::min());
    EXPECT_EQ(loadedRoot.getIntAttribute("int_max"), std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(loadedRoot.getFloatAttribute("float"), -0.1);
    EXPECT_EQ(loadedRoot.getStringAttribute("string"), "\xC3\xA4\xC3\xB6\xC3\xBC");
    EXPECT_EQ(loadedRoot.getStringAttribute("empty_string"), "");
    EXPECT_EQ(loadedRoot.getStringAttribute("replaced"), "replacement");
    EXPECT_TRUE(loadedRoot.hasAttribute("int_min"));
    EXPECT_FALSE(loadedRoot.hasAttribute("missing"));
}

TEST(Hdf5Test, enumerateAttributes) {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group group = file.getRootGroup().createGroup("group");
    EXPECT_TRUE(group.getAttributeNames().empty());
    group.setStringAttribute("c", "value");
    group.setIntAttribute("a", 1);
    group.setFloatAttribute("b", 1.0);

    std::vector<std::string> expectedNames = {"a", "b", "c"};
    EXPECT_EQ(group.getAttributeNames(), expectedNames);
    EXPECT_EQ(group.getAttributeTypeClass("a"), Hdf5TypeClass::INTEGER);
    EXPECT_EQ(group.getAttributeTypeClass("b"), Hdf5TypeClass::FLOAT);
    EXPECT_EQ(group.getAttributeTypeClass("c"), Hdf5TypeClass::STRING);
}

TEST(Hdf5Test, groupsAndPaths) {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group root = file.getRootGroup();
    Hdf5Group nested = root.createGroup("a/b/c");
    nested.setIntAttribute("depth", 3);
    std::vector<uint8> values = {1, 2};
    root.writeDataset<uint8>("a/d/dataset", values.data(), values.size());

    EXPECT_TRUE(root.hasGroup("a"));
    EXPECT_TRUE(root.hasGroup("a/b"));
    EXPECT_TRUE(root.hasGroup("a/b/c"));
    EXPECT_TRUE(root.hasGroup("a/d"));
    EXPECT_FALSE(root.hasGroup("a/d/dataset"));
    EXPECT_FALSE(root.hasGroup("missing"));
    EXPECT_FALSE(root.hasGroup("missing/b"));
    EXPECT_TRUE(root.hasDataset("a/d/dataset"));
    EXPECT_FALSE(root.hasDataset("a/b"));
    EXPECT_FALSE(root.hasDataset("a/d/missing"));
    EXPECT_FALSE(root.hasDataset("a/d/dataset/child"));

    std::vector<uint8> image = file.getImage();
    Hdf5File loadedFile = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group loadedRoot = loadedFile.getRootGroup();
    EXPECT_EQ(loadedRoot.openGroup("a").openGroup("b/c").getIntAttribute("depth"), 3);
    EXPECT_EQ(loadedRoot.openGroup("a/d").readDataset<uint8>("dataset"), values);
}

TEST(Hdf5Test, imageIsCopied) {
    std::vector<uint8> image = createImageWithContent();
    Hdf5File file = Hdf5File::openFromImage(image.data(), image.size());
    std::fill(image.begin(), image.end(), static_cast<uint8>(0));
    image.clear();
    image.shrink_to_fit();
    EXPECT_EQ(file.getRootGroup().getIntAttribute("int"), 42);
}

TEST(Hdf5Test, severalFilesCanBeOpenAtTheSameTime) {
    Hdf5File file1 = Hdf5File::createInMemory();
    Hdf5File file2 = Hdf5File::createInMemory();
    file1.getRootGroup().setIntAttribute("value", 1);
    file2.getRootGroup().setIntAttribute("value", 2);
    std::vector<uint8> image1 = file1.getImage();
    std::vector<uint8> image2 = file2.getImage();

    Hdf5File loadedFile1 = Hdf5File::openFromImage(image1.data(), image1.size());
    Hdf5File loadedFile2 = Hdf5File::openFromImage(image2.data(), image2.size());
    EXPECT_EQ(loadedFile1.getRootGroup().getIntAttribute("value"), 1);
    EXPECT_EQ(loadedFile2.getRootGroup().getIntAttribute("value"), 2);
}

TEST(Hdf5Test, openingInvalidImageThrows) {
    std::vector<uint8> garbage(1024, static_cast<uint8>(42));
    EXPECT_THROW(Hdf5File::openFromImage(garbage.data(), garbage.size()), std::runtime_error);
    EXPECT_THROW(Hdf5File::openFromImage(nullptr, 0), std::runtime_error);
    EXPECT_THROW(Hdf5File::openFromImage(garbage.data(), 0), std::runtime_error);

    std::vector<uint8> image = createImageWithContent();
    EXPECT_THROW(Hdf5File::openFromImage(image.data(), image.size() / 2), std::runtime_error);
}

TEST(Hdf5Test, accessingMissingObjectsThrows) {
    std::vector<uint8> image = createImageWithContent();
    Hdf5File file = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group root = file.getRootGroup();
    EXPECT_THROW(root.readDataset<uint32>("missing"), std::runtime_error);
    EXPECT_THROW(root.readStringDataset("missing"), std::runtime_error);
    EXPECT_THROW(root.getDatasetSize("missing"), std::runtime_error);
    EXPECT_THROW(root.openGroup("missing"), std::runtime_error);
    EXPECT_THROW(root.getIntAttribute("missing"), std::runtime_error);
    EXPECT_THROW(root.getAttributeTypeClass("missing"), std::runtime_error);
}

TEST(Hdf5Test, typeMismatchesThrow) {
    std::vector<uint8> image = createImageWithContent();
    Hdf5File file = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group root = file.getRootGroup();
    EXPECT_THROW(root.readDataset<uint8>("uint32"), std::runtime_error);
    EXPECT_THROW(root.readDataset<int32>("uint32"), std::runtime_error);
    EXPECT_THROW(root.readDataset<std::int64_t>("uint32"), std::runtime_error);
    EXPECT_THROW(root.readDataset<float32>("uint32"), std::runtime_error);
    EXPECT_THROW(root.readDataset<uint32>("strings"), std::runtime_error);
    EXPECT_THROW(root.readStringDataset("uint32"), std::runtime_error);
    EXPECT_THROW(root.getIntAttribute("float"), std::runtime_error);
    EXPECT_THROW(root.getIntAttribute("string"), std::runtime_error);
    EXPECT_THROW(root.getFloatAttribute("int"), std::runtime_error);
    EXPECT_THROW(root.getStringAttribute("int"), std::runtime_error);
    EXPECT_EQ(root.readDataset<uint32>("uint32"), std::vector<uint32>({1, 2, 3}));
}

TEST(Hdf5Test, writingExistingObjectsThrows) {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group root = file.getRootGroup();
    std::vector<uint8> values = {1};
    root.createGroup("group");
    root.writeDataset<uint8>("dataset", values.data(), values.size());
    EXPECT_THROW(root.createGroup("group"), std::runtime_error);
    EXPECT_THROW(root.writeDataset<uint8>("dataset", values.data(), values.size()), std::runtime_error);
    EXPECT_THROW(root.writeStringDataset("group", {"a"}), std::runtime_error);
}

TEST(Hdf5Test, writingToReadOnlyFileThrows) {
    std::vector<uint8> image = createImageWithContent();
    Hdf5File file = Hdf5File::openFromImage(image.data(), image.size());
    Hdf5Group root = file.getRootGroup();
    EXPECT_THROW(root.createGroup("new"), std::runtime_error);
    EXPECT_THROW(root.setIntAttribute("new", 1), std::runtime_error);
}

TEST(Hdf5Test, errorMessageContainsOperation) {
    Hdf5File file = Hdf5File::createInMemory();

    try {
        file.getRootGroup().openGroup("missing");
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("\"missing\""), std::string::npos) << error.what();
    }
}
