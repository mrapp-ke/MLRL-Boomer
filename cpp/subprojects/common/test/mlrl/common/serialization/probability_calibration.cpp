#include "mlrl/common/prediction/probability_calibration_isotonic.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"
#include "mlrl/common/serialization/hdf5/hdf5_file.hpp"
#include "mlrl/common/serialization/model_file.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

using Bin = std::tuple<uint32, float64, float64>;

static std::vector<Bin> getBins(const IIsotonicProbabilityCalibrationModel& model) {
    std::vector<Bin> bins;
    model.visit([&](uint32 listIndex, float64 threshold, float64 probability) {
        bins.emplace_back(listIndex, threshold, probability);
    });
    return bins;
}

static std::unique_ptr<IIsotonicProbabilityCalibrationModel> createIsotonicModel(uint32 numBinLists,
                                                                                 const std::vector<Bin>& bins) {
    std::unique_ptr<IIsotonicProbabilityCalibrationModel> modelPtr =
      createIsotonicProbabilityCalibrationModel(numBinLists);

    for (const Bin& bin : bins) {
        modelPtr->addBin(std::get<0>(bin), std::get<1>(bin), std::get<2>(bin));
    }

    return modelPtr;
}

static std::unique_ptr<IModelFileReader> writeAndRead(const IMarginalProbabilityCalibrationModel* marginalModel,
                                                      const IJointProbabilityCalibrationModel* jointModel) {
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();

    if (marginalModel) {
        writer->writeMarginalProbabilityCalibrationModel(*marginalModel);
    }

    if (jointModel) {
        writer->writeJointProbabilityCalibrationModel(*jointModel);
    }

    std::vector<uint8> data = writer->finish();
    return createModelFileReader(data.data(), data.size());
}

/**
 * Creates an HDF5 file with a valid header and a group for a marginal probability calibration model, whose content is
 * provided by a function.
 */
template<typename Function>
static std::unique_ptr<IModelFileReader> createCorruptedFile(Function function) {
    Hdf5File file = Hdf5File::createInMemory();
    Hdf5Group root = file.getRootGroup();
    root.setIntAttribute("format_version", MODEL_FILE_FORMAT_VERSION);
    root.setStringAttribute("library_version", "0.0.0");
    Hdf5Group group = root.createGroup("marginal_probability_calibration_model");
    function(group);
    std::vector<uint8> data = file.getImage();
    return createModelFileReader(data.data(), data.size());
}

static void writeIsotonicGroup(Hdf5Group& group, std::int64_t numBinLists, const std::vector<uint32>& offsets,
                               const std::vector<float64>& thresholds, const std::vector<float64>& probabilities) {
    group.setStringAttribute("type", "Isotonic");
    group.setIntAttribute("num_bin_lists", numBinLists);
    group.writeDataset<uint32>("offsets", offsets.data(), offsets.size());
    group.writeDataset<float64>("thresholds", thresholds.data(), thresholds.size());
    group.writeDataset<float64>("probabilities", probabilities.data(), probabilities.size());
}

/**
 * A model for the calibration of probabilities that is not supported by the model file writer.
 */
class UnsupportedProbabilityCalibrationModel final : public IMarginalProbabilityCalibrationModel,
                                                     public IJointProbabilityCalibrationModel {
    public:

        float64 calibrateMarginalProbability(uint32 labelIndex, float64 marginalProbability) const override {
            return marginalProbability;
        }

        float64 calibrateJointProbability(uint32 labelVectorIndex, float64 jointProbability) const override {
            return jointProbability;
        }
};

TEST(ProbabilityCalibrationSerializationTest, noProbabilityCalibrationModel) {
    std::unique_ptr<INoProbabilityCalibrationModel> modelPtr = createNoProbabilityCalibrationModel();
    std::unique_ptr<IModelFileReader> reader = writeAndRead(modelPtr.get(), modelPtr.get());
    ASSERT_TRUE(reader->hasMarginalProbabilityCalibrationModel());
    ASSERT_TRUE(reader->hasJointProbabilityCalibrationModel());

    std::unique_ptr<IMarginalProbabilityCalibrationModel> marginalModelPtr =
      reader->readMarginalProbabilityCalibrationModel();
    EXPECT_NE(dynamic_cast<INoProbabilityCalibrationModel*>(marginalModelPtr.get()), nullptr);
    EXPECT_EQ(marginalModelPtr->calibrateMarginalProbability(0, 0.3), 0.3);

    std::unique_ptr<IJointProbabilityCalibrationModel> jointModelPtr = reader->readJointProbabilityCalibrationModel();
    EXPECT_NE(dynamic_cast<INoProbabilityCalibrationModel*>(jointModelPtr.get()), nullptr);
    EXPECT_EQ(jointModelPtr->calibrateJointProbability(0, 0.7), 0.7);
}

TEST(ProbabilityCalibrationSerializationTest, isotonicModelWithoutBinLists) {
    std::unique_ptr<IIsotonicProbabilityCalibrationModel> modelPtr = createIsotonicModel(0, {});
    std::unique_ptr<IModelFileReader> reader = writeAndRead(modelPtr.get(), nullptr);
    std::unique_ptr<IMarginalProbabilityCalibrationModel> loadedModelPtr =
      reader->readMarginalProbabilityCalibrationModel();
    IIsotonicProbabilityCalibrationModel* isotonicModel =
      dynamic_cast<IIsotonicProbabilityCalibrationModel*>(loadedModelPtr.get());
    ASSERT_NE(isotonicModel, nullptr);
    EXPECT_EQ(isotonicModel->getNumBinLists(), (uint32) 0);
    EXPECT_TRUE(getBins(*isotonicModel).empty());
}

TEST(ProbabilityCalibrationSerializationTest, isotonicModelWithEmptyBinLists) {
    std::vector<Bin> bins = {Bin(1, 0.2, 0.1), Bin(1, 0.8, 0.9)};
    std::unique_ptr<IIsotonicProbabilityCalibrationModel> modelPtr = createIsotonicModel(3, bins);
    std::unique_ptr<IModelFileReader> reader = writeAndRead(nullptr, modelPtr.get());
    std::unique_ptr<IJointProbabilityCalibrationModel> loadedModelPtr = reader->readJointProbabilityCalibrationModel();
    IIsotonicProbabilityCalibrationModel* isotonicModel =
      dynamic_cast<IIsotonicProbabilityCalibrationModel*>(loadedModelPtr.get());
    ASSERT_NE(isotonicModel, nullptr);
    EXPECT_EQ(isotonicModel->getNumBinLists(), (uint32) 3);
    EXPECT_EQ(getBins(*isotonicModel), bins);
}

TEST(ProbabilityCalibrationSerializationTest, isotonicModelWithSingleBinList) {
    std::vector<Bin> bins = {Bin(0, 0.1, 0.05), Bin(0, 0.4, 0.3), Bin(0, 0.6, 0.65), Bin(0, 0.95, 0.99)};
    std::unique_ptr<IIsotonicProbabilityCalibrationModel> modelPtr = createIsotonicModel(1, bins);
    std::unique_ptr<IModelFileReader> reader = writeAndRead(modelPtr.get(), nullptr);
    EXPECT_FALSE(reader->hasJointProbabilityCalibrationModel());
    std::unique_ptr<IMarginalProbabilityCalibrationModel> loadedModelPtr =
      reader->readMarginalProbabilityCalibrationModel();
    IIsotonicProbabilityCalibrationModel* isotonicModel =
      dynamic_cast<IIsotonicProbabilityCalibrationModel*>(loadedModelPtr.get());
    ASSERT_NE(isotonicModel, nullptr);
    EXPECT_EQ(getBins(*isotonicModel), bins);

    for (float64 probability : {0.0, 0.05, 0.1, 0.25, 0.5, 0.6, 0.9, 0.95, 0.99, 1.0}) {
        EXPECT_EQ(loadedModelPtr->calibrateMarginalProbability(0, probability),
                  modelPtr->calibrateMarginalProbability(0, probability))
          << "probability = " << probability;
    }
}

TEST(ProbabilityCalibrationSerializationTest, isotonicModelWithSeveralBinLists) {
    std::vector<Bin> bins = {Bin(0, 0.1, 0.2), Bin(0, 0.9, 0.8), Bin(1, 0.5, 0.5),
                             Bin(2, 0.2, 0.1), Bin(2, 0.3, 0.4), Bin(2, 0.7, 0.75)};
    std::unique_ptr<IIsotonicProbabilityCalibrationModel> marginalModelPtr = createIsotonicModel(3, bins);
    std::unique_ptr<IIsotonicProbabilityCalibrationModel> jointModelPtr = createIsotonicModel(2, {Bin(1, 0.5, 0.6)});
    std::unique_ptr<IModelFileReader> reader = writeAndRead(marginalModelPtr.get(), jointModelPtr.get());

    std::unique_ptr<IMarginalProbabilityCalibrationModel> loadedMarginalModelPtr =
      reader->readMarginalProbabilityCalibrationModel();
    IIsotonicProbabilityCalibrationModel* loadedMarginalModel =
      dynamic_cast<IIsotonicProbabilityCalibrationModel*>(loadedMarginalModelPtr.get());
    ASSERT_NE(loadedMarginalModel, nullptr);
    EXPECT_EQ(loadedMarginalModel->getNumBinLists(), (uint32) 3);
    EXPECT_EQ(getBins(*loadedMarginalModel), bins);

    for (uint32 i = 0; i < 3; i++) {
        for (float64 probability : {0.0, 0.15, 0.5, 0.72, 1.0}) {
            EXPECT_EQ(loadedMarginalModelPtr->calibrateMarginalProbability(i, probability),
                      marginalModelPtr->calibrateMarginalProbability(i, probability))
              << "list = " << i << ", probability = " << probability;
        }
    }

    std::unique_ptr<IJointProbabilityCalibrationModel> loadedJointModelPtr =
      reader->readJointProbabilityCalibrationModel();
    IIsotonicProbabilityCalibrationModel* loadedJointModel =
      dynamic_cast<IIsotonicProbabilityCalibrationModel*>(loadedJointModelPtr.get());
    ASSERT_NE(loadedJointModel, nullptr);
    EXPECT_EQ(loadedJointModel->getNumBinLists(), (uint32) 2);
    EXPECT_EQ(getBins(*loadedJointModel), std::vector<Bin>({Bin(1, 0.5, 0.6)}));
    EXPECT_EQ(loadedJointModelPtr->calibrateJointProbability(1, 0.3), jointModelPtr->calibrateJointProbability(1, 0.3));
}

TEST(ProbabilityCalibrationSerializationTest, missingModelsAreReported) {
    std::unique_ptr<IModelFileReader> reader = writeAndRead(nullptr, nullptr);
    EXPECT_FALSE(reader->hasMarginalProbabilityCalibrationModel());
    EXPECT_FALSE(reader->hasJointProbabilityCalibrationModel());
    EXPECT_THROW(reader->readMarginalProbabilityCalibrationModel(), std::runtime_error);
    EXPECT_THROW(reader->readJointProbabilityCalibrationModel(), std::runtime_error);
}

TEST(ProbabilityCalibrationSerializationTest, writingUnsupportedModelThrows) {
    UnsupportedProbabilityCalibrationModel model;
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    EXPECT_THROW(writer->writeMarginalProbabilityCalibrationModel(model), std::runtime_error);
    EXPECT_THROW(writer->writeJointProbabilityCalibrationModel(model), std::runtime_error);

    // A failed attempt must not leave an incomplete group behind...
    std::vector<uint8> data = writer->finish();
    std::unique_ptr<IModelFileReader> reader = createModelFileReader(data.data(), data.size());
    EXPECT_FALSE(reader->hasMarginalProbabilityCalibrationModel());
    EXPECT_FALSE(reader->hasJointProbabilityCalibrationModel());
}

TEST(ProbabilityCalibrationSerializationTest, writingModelTwiceThrows) {
    std::unique_ptr<INoProbabilityCalibrationModel> modelPtr = createNoProbabilityCalibrationModel();
    std::unique_ptr<IModelFileWriter> writer = createModelFileWriter();
    writer->writeMarginalProbabilityCalibrationModel(*modelPtr);
    EXPECT_THROW(writer->writeMarginalProbabilityCalibrationModel(*modelPtr), std::runtime_error);
    writer->writeJointProbabilityCalibrationModel(*modelPtr);
    EXPECT_THROW(writer->writeJointProbabilityCalibrationModel(*modelPtr), std::runtime_error);
}

TEST(ProbabilityCalibrationSerializationTest, readingCorruptedModelThrows) {
    // Unknown type
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        group.setStringAttribute("type", "Unknown");
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Missing type
    EXPECT_THROW(createCorruptedFile([](Hdf5Group&) {
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Negative number of bin lists
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeIsotonicGroup(group, -1, {0}, {}, {});
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Number of offsets does not match the number of bin lists
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeIsotonicGroup(group, 2, {0, 1}, {0.5}, {0.5});
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // First offset is not zero
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeIsotonicGroup(group, 1, {1, 1}, {0.5}, {0.5});
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Offsets are not monotonic
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeIsotonicGroup(group, 2, {0, 2, 1}, {0.2, 0.5}, {0.2, 0.5});
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Intermediate offset exceeds the number of bins
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeIsotonicGroup(group, 2, {0, 3, 2}, {0.2, 0.5}, {0.2, 0.5});
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Last offset does not match the number of bins
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeIsotonicGroup(group, 1, {0, 3}, {0.2, 0.5}, {0.2, 0.5});
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Different number of thresholds and probabilities
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        writeIsotonicGroup(group, 1, {0, 2}, {0.2, 0.5}, {0.2});
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);

    // Missing dataset
    EXPECT_THROW(createCorruptedFile([](Hdf5Group& group) {
        group.setStringAttribute("type", "Isotonic");
        group.setIntAttribute("num_bin_lists", 0);
    })->readMarginalProbabilityCalibrationModel(),
                 std::runtime_error);
}
