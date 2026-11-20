#include "mlrl/common/model/rule_list.hpp"
#include "mlrl/common/prediction/label_vector_set.hpp"
#include "mlrl/common/prediction/output_space_info_no.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"
#include "mlrl/common/serialization/hdf5/hdf5_file.hpp"
#include "mlrl/common/serialization/model_file.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using LabelVectorEntry = std::pair<std::vector<uint32>, uint32>;

static std::vector<LabelVectorEntry> getLabelVectors(const ILabelVectorSet& labelVectorSet) {
    std::vector<LabelVectorEntry> entries;
    labelVectorSet.visit([&](const LabelVector& labelVector, uint32 frequency) {
        std::vector<uint32> labelIndices(labelVector.cbegin(), labelVector.cbegin() + labelVector.getNumElements());
        entries.emplace_back(labelIndices, frequency);
    });
    return entries;
}

static std::unique_ptr<ILabelVectorSet> buildLabelVectorSet(const std::vector<LabelVectorEntry>& entries) {
    std::unique_ptr<ILabelVectorSet> labelVectorSetPtr = createLabelVectorSet();

    for (const LabelVectorEntry& entry : entries) {
        const std::vector<uint32>& labelIndices = entry.first;
        std::unique_ptr<LabelVector> labelVectorPtr =
          std::make_unique<LabelVector>(static_cast<uint32>(labelIndices.size()));
        std::copy(labelIndices.begin(), labelIndices.end(), labelVectorPtr->begin());
        labelVectorSetPtr->addLabelVector(std::move(labelVectorPtr), entry.second);
    }

    return labelVectorSetPtr;
}

static std::unique_ptr<IModelFileReader> writeAndRead(const IOutputSpaceInfo& outputSpaceInfo) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeOutputSpaceInfo(outputSpaceInfo);
    std::vector<uint8> data = writer->finish();
    return createModelFileReader(data.data(), data.size());
}

static std::vector<LabelVectorEntry> writeAndReadLabelVectorSet(const std::vector<LabelVectorEntry>& entries) {
    std::unique_ptr<ILabelVectorSet> labelVectorSetPtr = buildLabelVectorSet(entries);
    std::unique_ptr<IModelFileReader> reader = writeAndRead(*labelVectorSetPtr);
    EXPECT_TRUE(reader->hasOutputSpaceInfo());
    std::unique_ptr<IOutputSpaceInfo> loadedPtr = reader->readOutputSpaceInfo();
    const ILabelVectorSet* loadedLabelVectorSet = dynamic_cast<const ILabelVectorSet*>(loadedPtr.get());

    if (!loadedLabelVectorSet) {
        ADD_FAILURE() << "Expected an object of type ILabelVectorSet";
        return {};
    }

    return getLabelVectors(*loadedLabelVectorSet);
}

/**
 * Creates an HDF5 file with a valid header and a group for information about the output space, whose content is
 * provided by a function.
 */
template<typename Function>
static std::unique_ptr<IModelFileReader> createCorruptedFile(Function function) {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group root = file.getRootGroup();
    root.setIntAttribute("format_version", MODEL_FILE_FORMAT_VERSION);
    root.setStringAttribute("library_version", "0.0.0");
    Hdf5Group group = root.createGroup("output_space_info");
    function(group);
    std::vector<uint8> data = file.getImage();
    return createModelFileReader(data.data(), data.size());
}

static void writeLabelVectorSetGroup(Hdf5Group& group, const std::vector<uint32>& offsets,
                                     const std::vector<uint32>& labelIndices, const std::vector<uint32>& frequencies) {
    group.setStringAttribute("type", "LabelVectorSet");
    group.writeDataset<uint32>("offsets", offsets.data(), offsets.size());
    group.writeDataset<uint32>("label_indices", labelIndices.data(), labelIndices.size());
    group.writeDataset<uint32>("frequencies", frequencies.data(), frequencies.size());
}

/**
 * Information about the output space that is not supported by the model file writer.
 */
class UnsupportedOutputSpaceInfo final : public IOutputSpaceInfo {
    public:

        std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator(
          const IJointProbabilityCalibratorFactory& factory,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override {
            return nullptr;
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousView<const float32>& featureMatrix,
                                                              const RuleList& model, uint32 numOutputs) const override {
            return nullptr;
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrView<const float32>& featureMatrix,
                                                              const RuleList& model, uint32 numOutputs) const override {
            return nullptr;
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return nullptr;
        }
};

TEST(OutputSpaceInfoSerializationTest, noOutputSpaceInfo) {
    std::unique_ptr<INoOutputSpaceInfo> outputSpaceInfoPtr = createNoOutputSpaceInfo();
    std::unique_ptr<IModelFileReader> reader = writeAndRead(*outputSpaceInfoPtr);
    ASSERT_TRUE(reader->hasOutputSpaceInfo());
    std::unique_ptr<IOutputSpaceInfo> loadedPtr = reader->readOutputSpaceInfo();
    EXPECT_NE(dynamic_cast<INoOutputSpaceInfo*>(loadedPtr.get()), nullptr);
    EXPECT_EQ(dynamic_cast<ILabelVectorSet*>(loadedPtr.get()), nullptr);
}

TEST(OutputSpaceInfoSerializationTest, emptyLabelVectorSet) {
    EXPECT_TRUE(writeAndReadLabelVectorSet({}).empty());
}

TEST(OutputSpaceInfoSerializationTest, labelVectorSetWithEmptyLabelVector) {
    std::vector<LabelVectorEntry> entries = {LabelVectorEntry({}, 7)};
    EXPECT_EQ(writeAndReadLabelVectorSet(entries), entries);

    entries = {LabelVectorEntry({1, 2}, 1), LabelVectorEntry({}, 3), LabelVectorEntry({0}, 2)};
    EXPECT_EQ(writeAndReadLabelVectorSet(entries), entries);
}

TEST(OutputSpaceInfoSerializationTest, labelVectorSetWithSeveralLabelVectors) {
    // The label vectors are deliberately not sorted, to ensure that their order is preserved
    std::vector<LabelVectorEntry> entries = {LabelVectorEntry({3, 5, 9}, 4), LabelVectorEntry({0}, 10),
                                             LabelVectorEntry({1, 2, 3, 4, 5, 6, 7, 8}, 1),
                                             LabelVectorEntry({2, 100000}, 2)};
    EXPECT_EQ(writeAndReadLabelVectorSet(entries), entries);
}

TEST(OutputSpaceInfoSerializationTest, labelVectorSetCanBeStoredWithOtherModels) {
    std::unique_ptr<ILabelVectorSet> labelVectorSetPtr = buildLabelVectorSet({LabelVectorEntry({0, 1}, 3)});
    std::unique_ptr<INoProbabilityCalibrationModel> calibrationModelPtr = createNoProbabilityCalibrationModel();
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeMarginalProbabilityCalibrationModel(*calibrationModelPtr);
    writer->writeOutputSpaceInfo(*labelVectorSetPtr);
    writer->writeJointProbabilityCalibrationModel(*calibrationModelPtr);
    std::vector<uint8> data = writer->finish();

    std::unique_ptr<IModelFileReader> reader = createModelFileReader(data.data(), data.size());
    EXPECT_TRUE(reader->hasMarginalProbabilityCalibrationModel());
    EXPECT_TRUE(reader->hasJointProbabilityCalibrationModel());
    std::unique_ptr<IOutputSpaceInfo> loadedPtr = reader->readOutputSpaceInfo();
    const ILabelVectorSet* loadedLabelVectorSet = dynamic_cast<const ILabelVectorSet*>(loadedPtr.get());
    ASSERT_NE(loadedLabelVectorSet, nullptr);
    EXPECT_EQ(getLabelVectors(*loadedLabelVectorSet), getLabelVectors(*labelVectorSetPtr));
}

TEST(OutputSpaceInfoSerializationTest, missingOutputSpaceInfoIsReported) {
    std::vector<uint8> data = createModelFileWriter()->finish();
    std::unique_ptr<IModelFileReader> reader = createModelFileReader(data.data(), data.size());
    EXPECT_FALSE(reader->hasOutputSpaceInfo());
    EXPECT_THROW(reader->readOutputSpaceInfo(), std::runtime_error);
}

TEST(OutputSpaceInfoSerializationTest, writingUnsupportedOutputSpaceInfoThrows) {
    UnsupportedOutputSpaceInfo outputSpaceInfo;
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    EXPECT_THROW(writer->writeOutputSpaceInfo(outputSpaceInfo), std::runtime_error);

    // A failed attempt must not leave an incomplete group behind...
    std::vector<uint8> data = writer->finish();
    std::unique_ptr<IModelFileReader> reader = createModelFileReader(data.data(), data.size());
    EXPECT_FALSE(reader->hasOutputSpaceInfo());
}

TEST(OutputSpaceInfoSerializationTest, writingOutputSpaceInfoTwiceThrows) {
    std::unique_ptr<INoOutputSpaceInfo> outputSpaceInfoPtr = createNoOutputSpaceInfo();
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeOutputSpaceInfo(*outputSpaceInfoPtr);
    EXPECT_THROW(writer->writeOutputSpaceInfo(*outputSpaceInfoPtr), std::runtime_error);
}

TEST(OutputSpaceInfoSerializationTest, readingCorruptedOutputSpaceInfoThrows) {
    // Unknown type
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        group.setStringAttribute("type", "Unknown");
    })->readOutputSpaceInfo(),
                 std::runtime_error);

    // Missing type
    EXPECT_THROW(createCorruptedFile([](Hdf5Group&) {
    })->readOutputSpaceInfo(),
                 std::runtime_error);

    // Number of offsets does not match the number of frequencies
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeLabelVectorSetGroup(group, {0, 1}, {0}, {1, 2});
    })->readOutputSpaceInfo(),
                 std::runtime_error);

    // First offset is not zero
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeLabelVectorSetGroup(group, {1, 1}, {0}, {1});
    })->readOutputSpaceInfo(),
                 std::runtime_error);

    // Offsets are not monotonic
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeLabelVectorSetGroup(group, {0, 2, 1}, {0}, {1, 1});
    })->readOutputSpaceInfo(),
                 std::runtime_error);

    // Intermediate offset exceeds the number of label indices
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeLabelVectorSetGroup(group, {0, 3, 2}, {0, 1}, {1, 1});
    })->readOutputSpaceInfo(),
                 std::runtime_error);

    // Last offset does not match the number of label indices
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeLabelVectorSetGroup(group, {0, 2}, {0}, {1});
    })->readOutputSpaceInfo(),
                 std::runtime_error);

    // Missing dataset
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        group.setStringAttribute("type", "LabelVectorSet");
        std::vector<uint32> offsets = {0};
        group.writeDataset<uint32>("offsets", offsets.data(), offsets.size());
    })->readOutputSpaceInfo(),
                 std::runtime_error);
}
