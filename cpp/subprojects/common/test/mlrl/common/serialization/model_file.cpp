#include "mlrl/common/serialization/model_file.hpp"

#include "mlrl/common/library_info.hpp"
#include "mlrl/common/serialization/hdf5/hdf5_file.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

static std::unique_ptr<IModelFileReader> createReader(const std::vector<uint8>& data) {
    return createModelFileReader(data.data(), data.size());
}

static std::vector<uint8> createHdf5FileWithFormatVersion(const std::int64_t* formatVersion) {
    Hdf5File file = Hdf5File::createInMemory();

    if (formatVersion) {
        file.getRootGroup().setIntAttribute("format_version", *formatVersion);
    }

    return file.getImage();
}

TEST(ModelFileTest, formatAndLibraryVersion) {
    std::vector<uint8> data = createModelFileWriter()->finish();
    std::unique_ptr<IModelFileReader> reader = createReader(data);
    EXPECT_EQ(reader->getFormatVersion(), MODEL_FILE_FORMAT_VERSION);
    EXPECT_EQ(reader->getLibraryVersion(), getLibraryInfo()->getLibraryVersion());
    EXPECT_TRUE(reader->hasGroup(""));
    EXPECT_TRUE(reader->hasGroup("/"));
    EXPECT_EQ(reader->getAttributeType("", "format_version"), ModelFileValueType::INT);
    EXPECT_EQ(reader->getAttributeType("/", "library_version"), ModelFileValueType::STRING);
}

TEST(ModelFileTest, attributeRoundTrip) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->setStringAttribute("learner", "class", "mlrl.boosting.BoomerClassifier");
    writer->setStringAttribute("learner/parameters", "unicode", "\xC3\xA4\xC3\xB6\xC3\xBC");
    writer->setIntAttribute("learner/parameters", "max_rules", 1000);
    writer->setIntAttribute("learner/parameters", "int_min", std::numeric_limits<std::int64_t>::min());
    writer->setIntAttribute("learner/parameters", "int_max", std::numeric_limits<std::int64_t>::max());
    writer->setFloatAttribute("learner/parameters", "shrinkage", 0.3);
    writer->setIntAttribute("/learner/parameters/", "replaced", 1);
    writer->setStringAttribute("learner/parameters", "replaced", "replacement");
    writer->setStringAttribute("", "custom", "root attribute");
    std::vector<uint8> data = writer->finish();

    std::unique_ptr<IModelFileReader> reader = createReader(data);
    EXPECT_EQ(reader->getStringAttribute("learner", "class"), "mlrl.boosting.BoomerClassifier");
    EXPECT_EQ(reader->getStringAttribute("learner/parameters", "unicode"), "\xC3\xA4\xC3\xB6\xC3\xBC");
    EXPECT_EQ(reader->getIntAttribute("learner/parameters", "max_rules"), 1000);
    EXPECT_EQ(reader->getIntAttribute("learner/parameters", "int_min"), std::numeric_limits<std::int64_t>::min());
    EXPECT_EQ(reader->getIntAttribute("learner/parameters", "int_max"), std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(reader->getFloatAttribute("learner/parameters", "shrinkage"), 0.3);
    EXPECT_EQ(reader->getStringAttribute("/learner/parameters", "replaced"), "replacement");
    EXPECT_EQ(reader->getStringAttribute("", "custom"), "root attribute");
}

TEST(ModelFileTest, attributeEnumeration) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->setFloatAttribute("parameters", "b", 1.0);
    writer->setStringAttribute("parameters", "c", "value");
    writer->setIntAttribute("parameters", "a", 1);
    writer->writeInt64Array("empty_group/subgroup", "array", nullptr, 0);
    std::vector<uint8> data = writer->finish();

    std::unique_ptr<IModelFileReader> reader = createReader(data);
    std::vector<std::string> expectedNames = {"a", "b", "c"};
    EXPECT_EQ(reader->getAttributeNames("parameters"), expectedNames);
    EXPECT_EQ(reader->getAttributeType("parameters", "a"), ModelFileValueType::INT);
    EXPECT_EQ(reader->getAttributeType("parameters", "b"), ModelFileValueType::FLOAT);
    EXPECT_EQ(reader->getAttributeType("parameters", "c"), ModelFileValueType::STRING);
    EXPECT_TRUE(reader->getAttributeNames("empty_group").empty());

    std::vector<std::string> expectedRootNames = {"format_version", "library_version"};
    EXPECT_EQ(reader->getAttributeNames(""), expectedRootNames);

    EXPECT_TRUE(reader->hasAttribute("parameters", "a"));
    EXPECT_FALSE(reader->hasAttribute("parameters", "missing"));
    EXPECT_FALSE(reader->hasAttribute("missing", "a"));
}

TEST(ModelFileTest, arrayRoundTrip) {
    std::vector<std::int64_t> intValues = {std::numeric_limits<std::int64_t>::min(), -1, 0, 1,
                                           std::numeric_limits<std::int64_t>::max()};
    std::vector<float64> floatValues = {-0.5, 0.0, 0.1, std::numeric_limits<float64>::infinity()};
    std::vector<std::string> stringValues = {"", "a", "\xE2\x82\xAC"};

    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeInt64Array("learner/fit_state", "ints", intValues.data(), intValues.size());
    writer->writeFloat64Array("learner/fit_state", "floats", floatValues.data(), floatValues.size());
    writer->writeStringArray("learner/fit_state", "strings", stringValues);
    writer->writeInt64Array("learner/fit_state", "empty_ints", nullptr, 0);
    writer->writeFloat64Array("learner/fit_state", "empty_floats", nullptr, 0);
    writer->writeStringArray("learner/fit_state", "empty_strings", {});
    writer->writeInt64Array("", "root_array", intValues.data(), 1);
    std::vector<uint8> data = writer->finish();

    std::unique_ptr<IModelFileReader> reader = createReader(data);
    EXPECT_EQ(reader->readInt64Array("learner/fit_state", "ints"), intValues);
    EXPECT_EQ(reader->readFloat64Array("learner/fit_state", "floats"), floatValues);
    EXPECT_EQ(reader->readStringArray("learner/fit_state", "strings"), stringValues);
    EXPECT_TRUE(reader->readInt64Array("learner/fit_state", "empty_ints").empty());
    EXPECT_TRUE(reader->readFloat64Array("learner/fit_state", "empty_floats").empty());
    EXPECT_TRUE(reader->readStringArray("learner/fit_state", "empty_strings").empty());
    EXPECT_EQ(reader->readInt64Array("/", "root_array"), std::vector<std::int64_t>({intValues[0]}));

    EXPECT_EQ(reader->getArrayType("learner/fit_state", "ints"), ModelFileValueType::INT);
    EXPECT_EQ(reader->getArrayType("learner/fit_state", "floats"), ModelFileValueType::FLOAT);
    EXPECT_EQ(reader->getArrayType("learner/fit_state", "strings"), ModelFileValueType::STRING);
    EXPECT_EQ(reader->getArrayType("learner/fit_state", "empty_strings"), ModelFileValueType::STRING);
}

TEST(ModelFileTest, groupsAndDatasets) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeInt64Array("a/b", "array", nullptr, 0);
    writer->setIntAttribute("c", "attribute", 1);
    std::vector<uint8> data = writer->finish();

    std::unique_ptr<IModelFileReader> reader = createReader(data);
    EXPECT_TRUE(reader->hasGroup("a"));
    EXPECT_TRUE(reader->hasGroup("a/b"));
    EXPECT_TRUE(reader->hasGroup("c"));
    EXPECT_FALSE(reader->hasGroup("a/b/array"));
    EXPECT_FALSE(reader->hasGroup("missing"));
    EXPECT_TRUE(reader->hasDataset("a/b", "array"));
    EXPECT_TRUE(reader->hasDataset("/a/b/", "array"));
    EXPECT_FALSE(reader->hasDataset("a", "b"));
    EXPECT_FALSE(reader->hasDataset("a/b", "missing"));
    EXPECT_FALSE(reader->hasDataset("missing", "array"));
}

TEST(ModelFileTest, dataIsCopied) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->setIntAttribute("group", "value", 42);
    std::vector<uint8> data = writer->finish();
    std::unique_ptr<IModelFileReader> reader = createReader(data);
    data.assign(data.size(), static_cast<uint8>(0));
    EXPECT_EQ(reader->getIntAttribute("group", "value"), 42);
}

TEST(ModelFileTest, writerCannotBeUsedAfterFinish) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->finish();
    std::int64_t value = 1;
    float64 floatValue = 1.0;
    EXPECT_THROW(writer->finish(), std::runtime_error);
    EXPECT_THROW(writer->setStringAttribute("group", "name", "value"), std::runtime_error);
    EXPECT_THROW(writer->setIntAttribute("group", "name", 1), std::runtime_error);
    EXPECT_THROW(writer->setFloatAttribute("group", "name", 1.0), std::runtime_error);
    EXPECT_THROW(writer->writeStringArray("group", "name", {"a"}), std::runtime_error);
    EXPECT_THROW(writer->writeInt64Array("group", "name", &value, 1), std::runtime_error);
    EXPECT_THROW(writer->writeFloat64Array("group", "name", &floatValue, 1), std::runtime_error);
}

TEST(ModelFileTest, reservedRootAttributesCannotBeOverwritten) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    EXPECT_THROW(writer->setIntAttribute("", "format_version", 2), std::runtime_error);
    EXPECT_THROW(writer->setStringAttribute("/", "library_version", "0.0.0"), std::runtime_error);
    writer->setIntAttribute("group", "format_version", 2);
    std::vector<uint8> data = writer->finish();
    std::unique_ptr<IModelFileReader> reader = createReader(data);
    EXPECT_EQ(reader->getFormatVersion(), MODEL_FILE_FORMAT_VERSION);
    EXPECT_EQ(reader->getIntAttribute("group", "format_version"), 2);
}

TEST(ModelFileTest, writingExistingArrayThrows) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    std::int64_t value = 1;
    writer->writeInt64Array("group", "array", &value, 1);
    EXPECT_THROW(writer->writeInt64Array("group", "array", &value, 1), std::runtime_error);
    EXPECT_THROW(writer->setIntAttribute("group/array", "name", 1), std::runtime_error);
}

TEST(ModelFileTest, invalidDataThrows) {
    std::vector<uint8> garbage(1024, static_cast<uint8>(7));
    EXPECT_THROW(createReader(garbage), std::runtime_error);
    EXPECT_THROW(createModelFileReader(nullptr, 0), std::runtime_error);

    std::vector<uint8> data = createModelFileWriter()->finish();
    EXPECT_THROW(createModelFileReader(data.data(), data.size() / 2), std::runtime_error);
}

TEST(ModelFileTest, missingFormatVersionThrows) {
    std::vector<uint8> data = createHdf5FileWithFormatVersion(nullptr);

    try {
        createReader(data);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("not a model file"), std::string::npos) << error.what();
    }
}

TEST(ModelFileTest, unsupportedFormatVersionThrows) {
    std::int64_t newerVersion = MODEL_FILE_FORMAT_VERSION + 1;
    std::vector<uint8> data = createHdf5FileWithFormatVersion(&newerVersion);

    try {
        createReader(data);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("format version " + std::to_string(newerVersion)), std::string::npos)
          << error.what();
    }

    std::int64_t invalidVersion = 0;
    data = createHdf5FileWithFormatVersion(&invalidVersion);
    EXPECT_THROW(createReader(data), std::runtime_error);
}

TEST(ModelFileTest, accessingMissingOrMismatchingValuesThrows) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->setIntAttribute("group", "int", 1);
    writer->writeFloat64Array("group", "floats", nullptr, 0);
    std::vector<uint8> data = writer->finish();

    std::unique_ptr<IModelFileReader> reader = createReader(data);
    EXPECT_THROW(reader->getIntAttribute("group", "missing"), std::runtime_error);
    EXPECT_THROW(reader->getIntAttribute("missing", "int"), std::runtime_error);
    EXPECT_THROW(reader->getAttributeNames("missing"), std::runtime_error);
    EXPECT_THROW(reader->getStringAttribute("group", "int"), std::runtime_error);
    EXPECT_THROW(reader->getFloatAttribute("group", "int"), std::runtime_error);
    EXPECT_THROW(reader->readInt64Array("group", "floats"), std::runtime_error);
    EXPECT_THROW(reader->readStringArray("group", "floats"), std::runtime_error);
    EXPECT_THROW(reader->readFloat64Array("group", "missing"), std::runtime_error);
    EXPECT_THROW(reader->getArrayType("group", "missing"), std::runtime_error);
}
