#include "mlrl/common/serialization/model_file.hpp"

#include "config.hpp"
#include "hdf5/hdf5_file.hpp"
#include "output_space_info_serialization.hpp"
#include "probability_calibration_serialization.hpp"
#include "rule_list_serialization.hpp"

#include <stdexcept>
#include <utility>

/**
 * The name of the attribute of the root group that stores the version of the file format.
 */
static const char ATTRIBUTE_FORMAT_VERSION[] = "format_version";

/**
 * The name of the attribute of the root group that stores the version of the library that has written the file.
 */
static const char ATTRIBUTE_LIBRARY_VERSION[] = "library_version";

/**
 * Removes leading and trailing separators from the path of a group. An empty string refers to the root group.
 */
static std::string normalizePath(const std::string& path) {
    std::size_t start = path.find_first_not_of('/');

    if (start == std::string::npos) {
        return std::string();
    }

    std::size_t end = path.find_last_not_of('/');
    return path.substr(start, end - start + 1);
}

static std::string joinPath(const std::string& group, const std::string& name) {
    std::string normalizedGroup = normalizePath(group);
    return normalizedGroup.empty() ? name : normalizedGroup + "/" + name;
}

static ModelFileValueType toValueType(Hdf5TypeClass typeClass, const std::string& description) {
    switch (typeClass) {
        case Hdf5TypeClass::INTEGER:
            return ModelFileValueType::INT;
        case Hdf5TypeClass::FLOAT:
            return ModelFileValueType::FLOAT;
        case Hdf5TypeClass::STRING:
            return ModelFileValueType::STRING;
        default:
            throw std::runtime_error(description + " has an unsupported type");
    }
}

/**
 * An implementation of the type `IModelFileWriter` that writes to an in-memory HDF5 file.
 */
class ModelFileWriter final : public IModelFileWriter {
    private:

        std::unique_ptr<Hdf5File> filePtr_;

        Hdf5File& getFile() {
            if (!filePtr_) {
                throw std::runtime_error("The model file has already been finished");
            }

            return *filePtr_;
        }

        Hdf5Group getOrCreateGroup(const std::string& group) {
            Hdf5Group root = this->getFile().getRootGroup();
            std::string path = normalizePath(group);

            if (path.empty()) {
                return root;
            }

            return root.hasGroup(path) ? root.openGroup(path) : root.createGroup(path);
        }

        Hdf5Group getGroupForAttribute(const std::string& group, const std::string& name) {
            if (normalizePath(group).empty()
                && (name == ATTRIBUTE_FORMAT_VERSION || name == ATTRIBUTE_LIBRARY_VERSION)) {
                throw std::runtime_error("The attribute \"" + name + "\" of the root group is reserved");
            }

            return this->getOrCreateGroup(group);
        }

    public:

        ModelFileWriter() : filePtr_(std::make_unique<Hdf5File>(Hdf5File::createInMemory())) {
            Hdf5Group root = filePtr_->getRootGroup();
            root.setIntAttribute(ATTRIBUTE_FORMAT_VERSION, MODEL_FILE_FORMAT_VERSION);
            root.setStringAttribute(ATTRIBUTE_LIBRARY_VERSION, MLRLCOMMON_LIBRARY_VERSION);
        }

        void setStringAttribute(const std::string& group, const std::string& name, const std::string& value) override {
            this->getGroupForAttribute(group, name).setStringAttribute(name, value);
        }

        void setIntAttribute(const std::string& group, const std::string& name, std::int64_t value) override {
            this->getGroupForAttribute(group, name).setIntAttribute(name, value);
        }

        void setFloatAttribute(const std::string& group, const std::string& name, float64 value) override {
            this->getGroupForAttribute(group, name).setFloatAttribute(name, value);
        }

        void writeStringArray(const std::string& group, const std::string& name,
                              const std::vector<std::string>& values) override {
            this->getOrCreateGroup(group).writeStringDataset(name, values);
        }

        void writeInt64Array(const std::string& group, const std::string& name, const std::int64_t* values,
                             std::size_t numValues) override {
            this->getOrCreateGroup(group).writeDataset<std::int64_t>(name, values, numValues);
        }

        void writeFloat64Array(const std::string& group, const std::string& name, const float64* values,
                               std::size_t numValues) override {
            this->getOrCreateGroup(group).writeDataset<float64>(name, values, numValues);
        }

        void writeMarginalProbabilityCalibrationModel(const IMarginalProbabilityCalibrationModel& model) override {
            Hdf5Group root = this->getFile().getRootGroup();
            ::writeMarginalProbabilityCalibrationModel(root, model);
        }

        void writeJointProbabilityCalibrationModel(const IJointProbabilityCalibrationModel& model) override {
            Hdf5Group root = this->getFile().getRootGroup();
            ::writeJointProbabilityCalibrationModel(root, model);
        }

        void writeOutputSpaceInfo(const IOutputSpaceInfo& outputSpaceInfo) override {
            Hdf5Group root = this->getFile().getRootGroup();
            ::writeOutputSpaceInfo(root, outputSpaceInfo);
        }

        void writeRuleModel(const IRuleModel& ruleModel) override {
            Hdf5Group root = this->getFile().getRootGroup();
            ::writeRuleModel(root, ruleModel);
        }

        std::vector<uint8> finish() override {
            std::vector<uint8> image = this->getFile().getImage();
            filePtr_.reset();
            return image;
        }
};

/**
 * An implementation of the type `IModelFileReader` that reads from an in-memory HDF5 file.
 */
class ModelFileReader final : public IModelFileReader {
    private:

        Hdf5File file_;

        Hdf5Group openGroup(const std::string& group) const {
            Hdf5Group root = file_.getRootGroup();
            std::string path = normalizePath(group);
            return path.empty() ? std::move(root) : root.openGroup(path);
        }

    public:

        ModelFileReader(const uint8* data, std::size_t size) : file_(Hdf5File::openFromImage(data, size)) {
            Hdf5Group root = file_.getRootGroup();

            if (!root.hasAttribute(ATTRIBUTE_FORMAT_VERSION)
                || root.getAttributeTypeClass(ATTRIBUTE_FORMAT_VERSION) != Hdf5TypeClass::INTEGER) {
                throw std::runtime_error("The given data is not a model file written by this library");
            }

            std::int64_t formatVersion = root.getIntAttribute(ATTRIBUTE_FORMAT_VERSION);

            if (formatVersion < 1 || formatVersion > MODEL_FILE_FORMAT_VERSION) {
                throw std::runtime_error("The model file uses the file format version " + std::to_string(formatVersion)
                                         + ", which is not supported by this version of the library (supported "
                                         + "versions are 1 to " + std::to_string(MODEL_FILE_FORMAT_VERSION)
                                         + "). Please use a newer version of the library");
            }
        }

        std::int64_t getFormatVersion() const override {
            return file_.getRootGroup().getIntAttribute(ATTRIBUTE_FORMAT_VERSION);
        }

        std::string getLibraryVersion() const override {
            return file_.getRootGroup().getStringAttribute(ATTRIBUTE_LIBRARY_VERSION);
        }

        bool hasGroup(const std::string& group) const override {
            std::string path = normalizePath(group);
            return path.empty() || file_.getRootGroup().hasGroup(path);
        }

        bool hasDataset(const std::string& group, const std::string& name) const override {
            return file_.getRootGroup().hasDataset(joinPath(group, name));
        }

        bool hasAttribute(const std::string& group, const std::string& name) const override {
            return this->hasGroup(group) && this->openGroup(group).hasAttribute(name);
        }

        std::vector<std::string> getAttributeNames(const std::string& group) const override {
            return this->openGroup(group).getAttributeNames();
        }

        ModelFileValueType getAttributeType(const std::string& group, const std::string& name) const override {
            return toValueType(this->openGroup(group).getAttributeTypeClass(name), "Attribute \"" + name + "\"");
        }

        std::string getStringAttribute(const std::string& group, const std::string& name) const override {
            return this->openGroup(group).getStringAttribute(name);
        }

        std::int64_t getIntAttribute(const std::string& group, const std::string& name) const override {
            return this->openGroup(group).getIntAttribute(name);
        }

        float64 getFloatAttribute(const std::string& group, const std::string& name) const override {
            return this->openGroup(group).getFloatAttribute(name);
        }

        ModelFileValueType getArrayType(const std::string& group, const std::string& name) const override {
            return toValueType(this->openGroup(group).getDatasetTypeClass(name), "Array \"" + name + "\"");
        }

        std::vector<std::string> readStringArray(const std::string& group, const std::string& name) const override {
            return this->openGroup(group).readStringDataset(name);
        }

        std::vector<std::int64_t> readInt64Array(const std::string& group, const std::string& name) const override {
            return this->openGroup(group).readDataset<std::int64_t>(name);
        }

        std::vector<float64> readFloat64Array(const std::string& group, const std::string& name) const override {
            return this->openGroup(group).readDataset<float64>(name);
        }

        bool hasMarginalProbabilityCalibrationModel() const override {
            return file_.getRootGroup().hasGroup(GROUP_MARGINAL_PROBABILITY_CALIBRATION_MODEL);
        }

        std::unique_ptr<IMarginalProbabilityCalibrationModel> readMarginalProbabilityCalibrationModel() const override {
            return ::readMarginalProbabilityCalibrationModel(file_.getRootGroup());
        }

        bool hasJointProbabilityCalibrationModel() const override {
            return file_.getRootGroup().hasGroup(GROUP_JOINT_PROBABILITY_CALIBRATION_MODEL);
        }

        std::unique_ptr<IJointProbabilityCalibrationModel> readJointProbabilityCalibrationModel() const override {
            return ::readJointProbabilityCalibrationModel(file_.getRootGroup());
        }

        bool hasOutputSpaceInfo() const override {
            return file_.getRootGroup().hasGroup(GROUP_OUTPUT_SPACE_INFO);
        }

        std::unique_ptr<IOutputSpaceInfo> readOutputSpaceInfo() const override {
            return ::readOutputSpaceInfo(file_.getRootGroup());
        }

        bool hasRuleModel() const override {
            return file_.getRootGroup().hasGroup(GROUP_RULE_MODEL);
        }

        std::unique_ptr<IRuleModel> readRuleModel() const override {
            return ::readRuleModel(file_.getRootGroup());
        }
};

std::unique_ptr<IModelFileWriter> createModelFileWriter() {
    return std::make_unique<ModelFileWriter>();
}

std::unique_ptr<IModelFileReader> createModelFileReader(const uint8* data, std::size_t size) {
    return std::make_unique<ModelFileReader>(data, size);
}
