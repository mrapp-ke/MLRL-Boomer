#include "mlrl/common/input/feature_type_nominal.hpp"

#include "feature_type_nominal_common.hpp"
#include "feature_vector_decorator_nominal.hpp"
#include "mlrl/common/iterator/iterator_index.hpp"

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<NominalFeatureVectorDecorator> createNominalFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numIndices, int32 majorityValue) {
    AllocatedNominalFeatureVector nominalFeatureVector(numValues, numIndices, majorityValue);
    AllocatedMissingFeatureVector missingFeatureVector;
    AllocatedNominalFeatureVector::value_iterator vectorValueIterator = nominalFeatureVector.values;
    AllocatedNominalFeatureVector::index_iterator vectorIndptrIterator = nominalFeatureVector.indptr;
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
            missingFeatureVector.set(index, true);
        } else {
            int32 nominalValue = static_cast<int32>(value);

            if (nominalValue != majorityValue) {
                Tuple<uint32>& tuple = mapping.at(nominalValue);
                uint32 numRemaining = tuple.second - 1;
                tuple.second = numRemaining;
                NominalFeatureVector::index_iterator vectorIndexIterator =
                  nominalFeatureVector.indices_begin(tuple.first);
                vectorIndexIterator[numRemaining] = index;
            }
        }
    }

    return std::make_unique<NominalFeatureVectorDecorator>(std::move(nominalFeatureVector),
                                                           std::move(missingFeatureVector));
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<NominalFeatureVectorDecorator> createNominalFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numExamples, bool sparse,
  int32 sparseValue) {
    int32 majorityValue;
    uint32 numMajorityExamples;

    if (sparse) {
        majorityValue = sparseValue;
        numMajorityExamples = 0;
    } else {
        majorityValue = getMajorityValue(mapping);
        numMajorityExamples = mapping.at(majorityValue).second;
    }

    return createNominalFeatureVector(indexIterator, valueIterator, numElements, mapping, numValues - 1,
                                      numExamples - numMajorityExamples, majorityValue);
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numExamples, bool sparse,
  int32 sparseValue) {
    if (numValues > 2) {
        return createNominalFeatureVector(indexIterator, valueIterator, numElements, mapping, numValues, numExamples,
                                          sparse, sparseValue);
    } else if (numValues > 1) {
        return createBinaryFeatureVector(indexIterator, valueIterator, numElements, mapping, sparse, sparseValue);
    } else {
        return std::make_unique<EqualFeatureVector>();
    }
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) {
    FortranContiguousView<const float32>::value_const_iterator valueIterator =
      featureMatrix.values_cbegin(featureIndex);
    uint32 numElements = featureMatrix.numRows;
    std::unordered_map<int32, Tuple<uint32>> mapping;
    uint32 numExamples = createMapping(valueIterator, numElements, mapping);
    uint32 numValues = static_cast<uint32>(mapping.size());
    return createFeatureVectorInternally(IndexIterator(), valueIterator, numElements, mapping, numValues, numExamples,
                                         false, 0);
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) {
    CscView<const float32>::index_const_iterator indexIterator = featureMatrix.indices_cbegin(featureIndex);
    CscView<const float32>::value_const_iterator valuesBegin = featureMatrix.values_cbegin(featureIndex);
    CscView<const float32>::value_const_iterator valuesEnd = featureMatrix.values_cend(featureIndex);
    uint32 numElements = valuesEnd - valuesBegin;
    std::unordered_map<int32, Tuple<uint32>> mapping;
    uint32 numExamples = createMapping(valuesBegin, numElements, mapping);
    uint32 numValues = static_cast<uint32>(mapping.size());
    bool sparse = numElements < featureMatrix.numRows;

    if (sparse) {
        numValues++;
    }

    return createFeatureVectorInternally(indexIterator, valuesBegin, numElements, mapping, numValues, numExamples,
                                         sparse, static_cast<int32>(featureMatrix.sparseValue));
}

std::unique_ptr<IFeatureVector> NominalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}

std::unique_ptr<IFeatureVector> NominalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}
