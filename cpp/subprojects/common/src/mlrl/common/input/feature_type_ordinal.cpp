#include "mlrl/common/input/feature_type_ordinal.hpp"

#include "feature_type_nominal_common.hpp"
#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/input/feature_vector_ordinal.hpp"
#include "mlrl/common/iterator/index_iterator.hpp"

#include <algorithm>

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indicies of
 * training examples stored in an `OrdinalFeatureVector`.
 */
class OrdinalFeatureVectorDecorator final : public AbstractNominalFeatureVectorDecorator {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        OrdinalFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                      AllocatedMissingFeatureVector&& secondView)
            : AbstractNominalFeatureVectorDecorator(std::move(firstView), std::move(secondView)) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end) const override {
            // TODO Implement
            return nullptr;
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            // TODO Implement
            return nullptr;
        }
};

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<OrdinalFeatureVectorDecorator> createOrdinalFeatureVector(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numIndices, int32 majorityValue) {
    DenseVector<int32> sortedValues(numValues);
    uint32 n = 0;

    for (auto it = mapping.cbegin(); it != mapping.cend(); it++) {
        auto& entry = *it;
        int32 value = entry.first;

        if (value != majorityValue) {
            sortedValues[n] = value;
            n++;
        }
    }

    std::sort(sortedValues.begin(), sortedValues.end(), std::less<int32>());

    AllocatedNominalFeatureVector ordinalFeatureVector(numValues, numIndices, majorityValue);
    AllocatedMissingFeatureVector missingFeatureVector;
    AllocatedNominalFeatureVector::value_iterator vectorValueIterator = ordinalFeatureVector.values;
    AllocatedNominalFeatureVector::index_iterator vectorIndptrIterator = ordinalFeatureVector.indptr;
    uint32 offset = 0;

    for (uint32 i = 0; i < numValues; i++) {
        int32 value = sortedValues[i];
        vectorValueIterator[i] = value;
        vectorIndptrIterator[i] = offset;
        Tuple<uint32>& tuple = mapping.at(value);
        tuple.first = i;
        offset += tuple.second;
    }

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];
        float32 value = valueIterator[i];

        if (std::isnan(value)) {
            missingFeatureVector.set(index, true);
        } else {
            int32 nominalValue = (int32) value;

            if (nominalValue != majorityValue) {
                Tuple<uint32>& tuple = mapping.at(nominalValue);
                uint32 numRemaining = tuple.second - 1;
                tuple.second = numRemaining;
                AllocatedNominalFeatureVector::index_iterator vectorIndexIterator =
                  ordinalFeatureVector.indices_begin(tuple.first);
                vectorIndexIterator[numRemaining] = index;
            }
        }
    }

    return std::make_unique<OrdinalFeatureVectorDecorator>(std::move(ordinalFeatureVector),
                                                           std::move(missingFeatureVector));
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<OrdinalFeatureVectorDecorator> createOrdinalFeatureVector(
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

    return createOrdinalFeatureVector(indexIterator, valueIterator, numElements, mapping, numValues - 1,
                                      numExamples - numMajorityExamples, majorityValue);
}

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  IndexIterator indexIterator, ValueIterator valueIterator, uint32 numElements,
  std::unordered_map<int32, Tuple<uint32>>& mapping, uint32 numValues, uint32 numExamples, bool sparse) {
    if (numValues > 2) {
        return createOrdinalFeatureVector(indexIterator, valueIterator, numElements, mapping, numValues, numExamples,
                                          sparse);
    } else if (numValues > 1) {
        return createBinaryFeatureVector(indexIterator, valueIterator, numElements, mapping, sparse);
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
    uint32 numValues = (uint32) mapping.size();
    return createFeatureVectorInternally(IndexIterator(), valueIterator, numElements, mapping, numValues, numExamples,
                                         false);
}

static inline std::unique_ptr<IFeatureVector> createFeatureVectorInternally(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) {
    CscView<const float32>::index_const_iterator indexIterator = featureMatrix.indices_cbegin(featureIndex);
    CscView<const float32>::value_const_iterator valuesBegin = featureMatrix.values_cbegin(featureIndex);
    CscView<const float32>::value_const_iterator valuesEnd = featureMatrix.values_cend(featureIndex);
    uint32 numElements = valuesEnd - valuesBegin;
    std::unordered_map<int32, Tuple<uint32>> mapping;
    uint32 numExamples = createMapping(valuesBegin, numElements, mapping);
    uint32 numValues = (uint32) mapping.size();
    bool sparse = numElements < featureMatrix.numRows;

    if (sparse) {
        numValues++;
    }

    return createFeatureVectorInternally(indexIterator, valuesBegin, numElements, mapping, numValues, numExamples,
                                         sparse);
}

bool OrdinalFeatureType::isOrdinal() const {
    return true;
}

bool OrdinalFeatureType::isNominal() const {
    return false;
}

std::unique_ptr<IFeatureVector> OrdinalFeatureType::createFeatureVector(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}

std::unique_ptr<IFeatureVector> OrdinalFeatureType::createFeatureVector(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) const {
    return createFeatureVectorInternally(featureIndex, featureMatrix);
}
