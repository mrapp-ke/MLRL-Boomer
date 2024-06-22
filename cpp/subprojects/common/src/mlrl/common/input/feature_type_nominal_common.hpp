/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_binary.hpp"
#include "mlrl/common/data/tuple.hpp"

#include <memory>
#include <unordered_map>
#include <utility>

template<typename ValueIterator>
static inline uint32 createMapping(ValueIterator valueIterator, uint32 numElements,
                                   std::unordered_map<int32, Tuple<uint32>>& mapping) {
    uint32 numExamples = 0;
    uint32 numValues = 0;

    for (uint32 i = 0; i < numElements; i++) {
        float32 value = valueIterator[i];

        if (!std::isnan(value)) {
            int32 nominalValue = static_cast<int32>(value);
            auto it = mapping.emplace(nominalValue, Tuple<uint32> {numValues, 1});

            if (it.second) {
                numValues++;
            } else {
                auto& entry = *(it.first);
                Tuple<uint32>& tuple = entry.second;
                tuple.second++;
            }

            numExamples++;
        }
    }

    return numExamples;
}

static inline int32 getMajorityValue(const std::unordered_map<int32, Tuple<uint32>>& mapping) {
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

static inline Tuple<int32> getMinorityAndMajorityValue(const std::unordered_map<int32, Tuple<uint32>>& mapping) {
    auto it = mapping.cbegin();
    auto& firstEntry = *it;
    int firstFrequency = firstEntry.second.second;
    it++;
    auto& secondEntry = *it;
    int secondFrequency = secondEntry.second.second;
    Tuple<int32> tuple;

    if (firstFrequency > secondFrequency) {
        tuple.first = secondEntry.first;
        tuple.second = firstEntry.first;
    } else {
        tuple.first = firstEntry.first;
        tuple.second = secondEntry.first;
    }

    return tuple;
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<BinaryFeatureVectorDecorator> createBinaryFeatureVectorInternally(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  const std::unordered_map<int32, Tuple<uint32>>& mapping, int32 minorityValue, int32 majorityValue) {
    const Tuple<uint32>& tuple = mapping.at(minorityValue);
    uint32 numMinorityExamples = tuple.second;
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
  std::unordered_map<int32, Tuple<uint32>>& mapping, bool sparse, int32 sparseValue) {
    int32 minorityValue;
    int32 majorityValue;

    if (sparse) {
        minorityValue = (*mapping.cbegin()).first;
        majorityValue = sparseValue;
    } else {
        const Tuple<int32> tuple = getMinorityAndMajorityValue(mapping);
        minorityValue = tuple.first;
        majorityValue = tuple.second;
    }

    return createBinaryFeatureVectorInternally(indexIterator, valueIterator, numElements, mapping, minorityValue,
                                               majorityValue);
}
