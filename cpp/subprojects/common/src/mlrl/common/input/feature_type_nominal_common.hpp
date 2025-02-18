/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_binary.hpp"

#include <memory>
#include <unordered_map>
#include <utility>

template<typename ValueIterator>
static inline uint32 createMapping(ValueIterator valueIterator, uint32 numElements,
                                   std::unordered_map<int32, std::pair<uint32, uint32>>& mapping) {
    uint32 numExamples = 0;
    uint32 numValues = 0;

    for (uint32 i = 0; i < numElements; i++) {
        float32 value = valueIterator[i];

        if (!std::isnan(value)) {
            int32 nominalValue = static_cast<int32>(value);
            auto it = mapping.emplace(nominalValue, std::make_pair(numValues, 1));

            if (it.second) {
                numValues++;
            } else {
                auto& entry = *(it.first);
                std::pair<uint32, uint32>& pair = entry.second;
                pair.second++;
            }

            numExamples++;
        }
    }

    return numExamples;
}

static inline int32 getMajorityValue(const std::unordered_map<int32, std::pair<uint32, uint32>>& mapping) {
    auto it = mapping.cbegin();
    auto& firstEntry = *it;
    int32 majorityValue = firstEntry.first;
    uint32 majorityValueFrequency = firstEntry.second.second;

    for (; it != mapping.cend(); it++) {
        auto& entry = *it;
        uint32 frequency = entry.second.second;

        if (frequency > majorityValueFrequency) {
            majorityValue = entry.first;
            majorityValueFrequency = frequency;
        }
    }

    return majorityValue;
}

static inline std::pair<int32, int32> getMinorityAndMajorityValue(
  const std::unordered_map<int32, std::pair<uint32, uint32>>& mapping) {
    auto it = mapping.cbegin();
    auto& firstEntry = *it;
    int firstFrequency = firstEntry.second.second;
    it++;
    auto& secondEntry = *it;
    int secondFrequency = secondEntry.second.second;
    int32 minorityValue;
    int32 majorityValue;

    if (firstFrequency > secondFrequency) {
        minorityValue = secondEntry.first;
        majorityValue = firstEntry.first;
    } else {
        minorityValue = firstEntry.first;
        majorityValue = secondEntry.first;
    }

    return std::make_pair(minorityValue, majorityValue);
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<BinaryFeatureVectorDecorator> createBinaryFeatureVectorInternally(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  const std::unordered_map<int32, std::pair<uint32, uint32>>& mapping, int32 minorityValue, int32 majorityValue) {
    const std::pair<uint32, uint32>& pair = mapping.at(minorityValue);
    uint32 numMinorityExamples = pair.second;
    AllocatedNominalFeatureVector binaryFeatureVector(1, numMinorityExamples, majorityValue);
    AllocatedMissingFeatureVector missingFeatureVector;
    AllocatedNominalFeatureVector::value_iterator vectorValueIterator = binaryFeatureVector.values;
    vectorValueIterator[0] = minorityValue;
    AllocatedNominalFeatureVector::index_iterator vectorIndexIterator = binaryFeatureVector.indices;
    uint32 n = 0;

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];
        float32 value = valueIterator[i];

        if (std::isnan(value)) {
            missingFeatureVector.set(index, true);
        } else {
            int32 nominalValue = static_cast<int32>(value);

            if (nominalValue == minorityValue) {
                vectorIndexIterator[n] = index;
                n++;
            }
        }
    }

    return std::make_unique<BinaryFeatureVectorDecorator>(std::move(binaryFeatureVector),
                                                          std::move(missingFeatureVector));
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<BinaryFeatureVectorDecorator> createBinaryFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, std::pair<uint32, uint32>>& mapping, bool sparse, int32 sparseValue) {
    int32 minorityValue;
    int32 majorityValue;

    if (sparse) {
        minorityValue = (*mapping.cbegin()).first;
        majorityValue = sparseValue;
    } else {
        const std::pair<int32, int32> pair = getMinorityAndMajorityValue(mapping);
        minorityValue = pair.first;
        majorityValue = pair.second;
    }

    return createBinaryFeatureVectorInternally(indexIterator, valueIterator, numElements, mapping, minorityValue,
                                               majorityValue);
}
