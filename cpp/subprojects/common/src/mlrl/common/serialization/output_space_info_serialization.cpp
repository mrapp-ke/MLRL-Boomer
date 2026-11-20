#include "output_space_info_serialization.hpp"

#include "mlrl/common/prediction/label_vector_set.hpp"
#include "mlrl/common/prediction/output_space_info_no.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static const char ATTRIBUTE_TYPE[] = "type";
static const char DATASET_OFFSETS[] = "offsets";
static const char DATASET_LABEL_INDICES[] = "label_indices";
static const char DATASET_FREQUENCIES[] = "frequencies";
static const char TYPE_NONE[] = "None";
static const char TYPE_LABEL_VECTOR_SET[] = "LabelVectorSet";

[[noreturn]] static void throwInconsistentData() {
    throw std::runtime_error("Group \"" + std::string(GROUP_OUTPUT_SPACE_INFO)
                             + "\" stores inconsistent information about the output space");
}

static void writeLabelVectorSet(Hdf5Group& parent, const ILabelVectorSet& labelVectorSet) {
    std::vector<uint32> offsets = {0};
    std::vector<uint32> labelIndices;
    std::vector<uint32> frequencies;
    labelVectorSet.visit([&](const LabelVector& labelVector, uint32 frequency) {
        uint32 numElements = labelVector.getNumElements();
        LabelVector::const_iterator iterator = labelVector.cbegin();

        for (uint32 i = 0; i < numElements; i++) {
            labelIndices.push_back(iterator[i]);
        }

        if (labelIndices.size() > std::numeric_limits<uint32>::max()) {
            throw std::runtime_error("Label vector set contains too many label indices");
        }

        offsets.push_back(static_cast<uint32>(labelIndices.size()));
        frequencies.push_back(frequency);
    });

    Hdf5Group group = parent.createGroup(GROUP_OUTPUT_SPACE_INFO);
    group.setStringAttribute(ATTRIBUTE_TYPE, TYPE_LABEL_VECTOR_SET);
    group.writeDataset<uint32>(DATASET_OFFSETS, offsets.data(), offsets.size());
    group.writeDataset<uint32>(DATASET_LABEL_INDICES, labelIndices.data(), labelIndices.size());
    group.writeDataset<uint32>(DATASET_FREQUENCIES, frequencies.data(), frequencies.size());
}

static std::unique_ptr<ILabelVectorSet> readLabelVectorSet(const Hdf5Group& group) {
    std::vector<uint32> offsets = group.readDataset<uint32>(DATASET_OFFSETS);
    std::vector<uint32> labelIndices = group.readDataset<uint32>(DATASET_LABEL_INDICES);
    std::vector<uint32> frequencies = group.readDataset<uint32>(DATASET_FREQUENCIES);
    std::size_t numLabelVectors = frequencies.size();

    if (offsets.size() != numLabelVectors + 1 || offsets[0] != 0 || offsets[numLabelVectors] != labelIndices.size()) {
        throwInconsistentData();
    }

    std::unique_ptr<ILabelVectorSet> labelVectorSetPtr = createLabelVectorSet();

    for (std::size_t i = 0; i < numLabelVectors; i++) {
        uint32 start = offsets[i];
        uint32 end = offsets[i + 1];

        if (start > end || end > labelIndices.size()) {
            throwInconsistentData();
        }

        std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(end - start);
        LabelVector::iterator iterator = labelVectorPtr->begin();

        for (uint32 j = start; j < end; j++) {
            iterator[j - start] = labelIndices[j];
        }

        labelVectorSetPtr->addLabelVector(std::move(labelVectorPtr), frequencies[i]);
    }

    return labelVectorSetPtr;
}

void writeOutputSpaceInfo(Hdf5Group& parent, const IOutputSpaceInfo& outputSpaceInfo) {
    if (const ILabelVectorSet* labelVectorSet = dynamic_cast<const ILabelVectorSet*>(&outputSpaceInfo)) {
        writeLabelVectorSet(parent, *labelVectorSet);
    } else if (dynamic_cast<const INoOutputSpaceInfo*>(&outputSpaceInfo)) {
        Hdf5Group group = parent.createGroup(GROUP_OUTPUT_SPACE_INFO);
        group.setStringAttribute(ATTRIBUTE_TYPE, TYPE_NONE);
    } else {
        throw std::runtime_error("Unable to write information about the output space of unsupported type");
    }
}

std::unique_ptr<IOutputSpaceInfo> readOutputSpaceInfo(const Hdf5Group& parent) {
    Hdf5Group group = parent.openGroup(GROUP_OUTPUT_SPACE_INFO);
    std::string type = group.getStringAttribute(ATTRIBUTE_TYPE);

    if (type == TYPE_NONE) {
        return createNoOutputSpaceInfo();
    } else if (type == TYPE_LABEL_VECTOR_SET) {
        return readLabelVectorSet(group);
    }

    throw std::runtime_error("Group \"" + std::string(GROUP_OUTPUT_SPACE_INFO)
                             + "\" stores information about the output space of unknown type \"" + type + "\"");
}
