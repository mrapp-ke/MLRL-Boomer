#include "mlrl/common/input/feature_type_nominal.hpp"

#include "mlrl/common/data/tuple.hpp"
#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"
#include "mlrl/common/iterator/index_iterator.hpp"

#include <iostream>
#include <unordered_map>

template<typename ValueIterator>
static inline uint32 createMapping(ValueIterator valueIterator, uint32 numElements,
                                   std::unordered_map<int32, Tuple<uint32>>& mapping) {
    uint32 numExamples = 0;
    uint32 numValues = 0;

    for (uint32 i = 0; i < numElements; i++) {
        float32 value = valueIterator[i];

        if (!std::isnan(value)) {
            int32 nominalValue = (int32) value;
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

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<NominalFeatureVector> createNominalFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numExamples, int32 majorityValue) {
    std::unique_ptr<NominalFeatureVector> featureVectorPtr =
      std::make_unique<NominalFeatureVector>(numValues, numExamples, majorityValue);
    NominalFeatureVector::value_iterator vectorValueIterator = featureVectorPtr->values_begin();
    NominalFeatureVector::index_iterator vectorIndptrIterator = featureVectorPtr->indptr_begin();
    uint32 offset = 0;
    uint32 n = 0;

    for (auto it = mapping.begin(); it != mapping.end(); it++) {
        auto& entry = *it;
        int32 value = entry.first;

        if (value != majorityValue) {
            vectorValueIterator[n] = value;
            vectorIndptrIterator[n] = offset;
            Tuple<uint32>& tuple = entry.second;
            tuple.first = n;
            offset += tuple.second;
            n++;
        }
    }

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];
        float32 value = valueIterator[i];

        if (std::isnan(value)) {
            featureVectorPtr->setMissing(index, true);
        } else {
            int32 nominalValue = (int32) value;

            if (nominalValue != majorityValue) {
                Tuple<uint32>& tuple = mapping.at(nominalValue);
                uint32 numRemaining = tuple.second - 1;
                tuple.second = numRemaining;
                NominalFeatureVector::index_iterator vectorIndexIterator = featureVectorPtr->indices_begin(tuple.first);
                vectorIndexIterator[numRemaining] = index;
            }
        }
    }

    return featureVectorPtr;
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<NominalFeatureVector> createNominalFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numExamples, bool sparse) {
    int32 majorityValue;
    uint32 numMajorityExamples;

    if (sparse) {
        majorityValue = 0;
        numMajorityExamples = 0;
    } else {
        majorityValue = getMajorityValue(mapping);
        numMajorityExamples = mapping.at(majorityValue).second;
    }

    return createNominalFeatureVector(indexIterator, valueIterator, numElements, mapping, numValues - 1,
                                      numExamples - numMajorityExamples, majorityValue);
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<BinaryFeatureVector> createBinaryFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  const std::unordered_map<int32, Tuple<uint32>>& mapping, int32 minorityValue, int32 majorityValue) {
    const Tuple<uint32>& tuple = mapping.at(minorityValue);
    uint32 numMinorityExamples = tuple.second;
    std::unique_ptr<BinaryFeatureVector> featureVectorPtr =
      std::make_unique<BinaryFeatureVector>(numMinorityExamples, minorityValue, majorityValue);
    BinaryFeatureVector::index_iterator vectorIndexIterator = featureVectorPtr->indices_begin(0);
    uint32 n = 0;

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];
        float32 value = valueIterator[i];

        if (std::isnan(value)) {
            featureVectorPtr->setMissing(index, true);
        } else {
            int32 nominalValue = (int32) value;

            if (nominalValue == minorityValue) {
                vectorIndexIterator[n] = index;
                n++;
            }
        }
    }

    return featureVectorPtr;
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<BinaryFeatureVector> createBinaryFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, bool sparse) {
    int32 minorityValue;
    int32 majorityValue;

    if (sparse) {
        minorityValue = (*mapping.cbegin()).first;
        majorityValue = 0;
    } else {
        const Tuple<int32> tuple = getMinorityAndMajorityValue(mapping);
        minorityValue = tuple.first;
        majorityValue = tuple.second;
    }

    return createBinaryFeatureVector(indexIterator, valueIterator, numElements, mapping, minorityValue, majorityValue);
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numExamples, bool sparse) {
    if (numValues > 2) {
        return createNominalFeatureVector(indexIterator, valueIterator, numElements, mapping, numValues, numExamples,
                                          sparse);
    } else if (numValues > 1) {
        return createBinaryFeatureVector(indexIterator, valueIterator, numElements, mapping, sparse);
    } else {
        return std::make_unique<EqualFeatureVector>();
    }
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const FortranContiguousConstView<const float32>& featureMatrix) {
    FortranContiguousConstView<const float32>::value_const_iterator valueIterator =
      featureMatrix.values_cbegin(featureIndex);
    uint32 numElements = featureMatrix.getNumRows();
    std::unordered_map<int32, Tuple<uint32>> mapping;
    uint32 numExamples = createMapping(valueIterator, numElements, mapping);
    uint32 numValues = (uint32) mapping.size();
    return createFeatureVectorInternally(IndexIterator(), valueIterator, numElements, mapping, numValues, numExamples,
                                         false);
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const CscConstView<const float32>& featureMatrix) {
    CscConstView<const float32>::index_const_iterator indexIterator = featureMatrix.indices_cbegin(featureIndex);
    CscConstView<const float32>::value_const_iterator valuesBegin = featureMatrix.values_cbegin(featureIndex);
    CscConstView<const float32>::value_const_iterator valuesEnd = featureMatrix.values_cend(featureIndex);
    uint32 numElements = valuesEnd - valuesBegin;
    std::unordered_map<int32, Tuple<uint32>> mapping;
    uint32 numExamples = createMapping(valuesBegin, numElements, mapping);
    uint32 numValues = (uint32) mapping.size();
    bool sparse = numElements < featureMatrix.getNumRows();

    if (sparse) {
        numValues++;
    }

    return createFeatureVectorInternally(indexIterator, valuesBegin, numElements, mapping, numValues, numExamples,
                                         sparse);
}

bool NominalFeatureType::isOrdinal() const {
    return false;
}

bool NominalFeatureType::isNominal() const {
    return true;
}

std::unique_ptr<IFeatureVector> NominalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousConstView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}

std::unique_ptr<IFeatureVector> NominalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscConstView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}
