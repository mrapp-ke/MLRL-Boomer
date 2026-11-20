#include "probability_calibration_serialization.hpp"

#include "mlrl/common/prediction/probability_calibration_isotonic.hpp"
#include "mlrl/common/prediction/probability_calibration_no.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static const char ATTRIBUTE_TYPE[] = "type";
static const char ATTRIBUTE_NUM_BIN_LISTS[] = "num_bin_lists";
static const char DATASET_OFFSETS[] = "offsets";
static const char DATASET_THRESHOLDS[] = "thresholds";
static const char DATASET_PROBABILITIES[] = "probabilities";
static const char TYPE_NONE[] = "None";
static const char TYPE_ISOTONIC[] = "Isotonic";

/**
 * Writes a model for the calibration of probabilities to a new group. The model may either be of type
 * `INoProbabilityCalibrationModel` or `IIsotonicProbabilityCalibrationModel`, i.e., exactly one of the given pointers
 * must not be a null pointer.
 */
static void writeProbabilityCalibrationModel(Hdf5Group& parent, const std::string& groupName,
                                             const INoProbabilityCalibrationModel* noModel,
                                             const IIsotonicProbabilityCalibrationModel* isotonicModel) {
    if (noModel) {
        Hdf5Group group = parent.createGroup(groupName);
        group.setStringAttribute(ATTRIBUTE_TYPE, TYPE_NONE);
    } else if (isotonicModel) {
        // Collect the bins per list, regardless of the order in which they are visited...
        uint32 numBinLists = isotonicModel->getNumBinLists();
        std::vector<std::vector<std::pair<float64, float64>>> binLists(numBinLists);
        isotonicModel->visit([&](uint32 listIndex, float64 threshold, float64 probability) {
            if (listIndex >= numBinLists) {
                throw std::runtime_error("Probability calibration model contains a bin with invalid list index "
                                         + std::to_string(listIndex));
            }

            binLists[listIndex].emplace_back(threshold, probability);
        });

        std::vector<uint32> offsets;
        offsets.reserve(numBinLists + 1);
        offsets.push_back(0);
        std::vector<float64> thresholds;
        std::vector<float64> probabilities;

        for (const std::vector<std::pair<float64, float64>>& binList : binLists) {
            for (const std::pair<float64, float64>& bin : binList) {
                thresholds.push_back(bin.first);
                probabilities.push_back(bin.second);
            }

            if (thresholds.size() > std::numeric_limits<uint32>::max()) {
                throw std::runtime_error("Probability calibration model contains too many bins");
            }

            offsets.push_back(static_cast<uint32>(thresholds.size()));
        }

        Hdf5Group group = parent.createGroup(groupName);
        group.setStringAttribute(ATTRIBUTE_TYPE, TYPE_ISOTONIC);
        group.setIntAttribute(ATTRIBUTE_NUM_BIN_LISTS, numBinLists);
        group.writeDataset<uint32>(DATASET_OFFSETS, offsets.data(), offsets.size());
        group.writeDataset<float64>(DATASET_THRESHOLDS, thresholds.data(), thresholds.size());
        group.writeDataset<float64>(DATASET_PROBABILITIES, probabilities.data(), probabilities.size());
    } else {
        throw std::runtime_error("Unable to write probability calibration model of unsupported type");
    }
}

/**
 * Reads a model for the calibration of probabilities from a group. The returned model implements the interfaces
 * `IMarginalProbabilityCalibrationModel` as well as `IJointProbabilityCalibrationModel`.
 */
template<typename Model>
static std::unique_ptr<Model> readProbabilityCalibrationModel(const Hdf5Group& parent, const std::string& groupName) {
    Hdf5Group group = parent.openGroup(groupName);
    std::string type = group.getStringAttribute(ATTRIBUTE_TYPE);

    if (type == TYPE_NONE) {
        return createNoProbabilityCalibrationModel();
    } else if (type != TYPE_ISOTONIC) {
        throw std::runtime_error("Group \"" + groupName + "\" stores a probability calibration model of unknown type \""
                                 + type + "\"");
    }

    std::int64_t numBinLists = group.getIntAttribute(ATTRIBUTE_NUM_BIN_LISTS);

    if (numBinLists < 0 || numBinLists > std::numeric_limits<uint32>::max()) {
        throw std::runtime_error("Group \"" + groupName + "\" specifies an invalid number of bin lists");
    }

    std::vector<uint32> offsets = group.readDataset<uint32>(DATASET_OFFSETS);
    std::vector<float64> thresholds = group.readDataset<float64>(DATASET_THRESHOLDS);
    std::vector<float64> probabilities = group.readDataset<float64>(DATASET_PROBABILITIES);
    std::size_t numLists = static_cast<std::size_t>(numBinLists);

    if (offsets.size() != numLists + 1 || offsets[0] != 0 || offsets[numLists] != thresholds.size()
        || thresholds.size() != probabilities.size()) {
        throw std::runtime_error("Group \"" + groupName + "\" stores an inconsistent probability calibration model");
    }

    std::unique_ptr<IIsotonicProbabilityCalibrationModel> modelPtr =
      createIsotonicProbabilityCalibrationModel(static_cast<uint32>(numLists));

    for (std::size_t i = 0; i < numLists; i++) {
        uint32 start = offsets[i];
        uint32 end = offsets[i + 1];

        if (start > end || end > thresholds.size()) {
            throw std::runtime_error("Group \"" + groupName
                                     + "\" stores an inconsistent probability calibration model");
        }

        for (uint32 j = start; j < end; j++) {
            modelPtr->addBin(static_cast<uint32>(i), thresholds[j], probabilities[j]);
        }
    }

    return modelPtr;
}

void writeMarginalProbabilityCalibrationModel(Hdf5Group& parent, const IMarginalProbabilityCalibrationModel& model) {
    writeProbabilityCalibrationModel(parent, GROUP_MARGINAL_PROBABILITY_CALIBRATION_MODEL,
                                     dynamic_cast<const INoProbabilityCalibrationModel*>(&model),
                                     dynamic_cast<const IIsotonicProbabilityCalibrationModel*>(&model));
}

void writeJointProbabilityCalibrationModel(Hdf5Group& parent, const IJointProbabilityCalibrationModel& model) {
    writeProbabilityCalibrationModel(parent, GROUP_JOINT_PROBABILITY_CALIBRATION_MODEL,
                                     dynamic_cast<const INoProbabilityCalibrationModel*>(&model),
                                     dynamic_cast<const IIsotonicProbabilityCalibrationModel*>(&model));
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> readMarginalProbabilityCalibrationModel(const Hdf5Group& parent) {
    return readProbabilityCalibrationModel<IMarginalProbabilityCalibrationModel>(
      parent, GROUP_MARGINAL_PROBABILITY_CALIBRATION_MODEL);
}

std::unique_ptr<IJointProbabilityCalibrationModel> readJointProbabilityCalibrationModel(const Hdf5Group& parent) {
    return readProbabilityCalibrationModel<IJointProbabilityCalibrationModel>(
      parent, GROUP_JOINT_PROBABILITY_CALIBRATION_MODEL);
}
