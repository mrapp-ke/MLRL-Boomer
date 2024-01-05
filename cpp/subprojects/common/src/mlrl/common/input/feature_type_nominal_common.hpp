/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_type_common.hpp"
#include "feature_vector_nominal_allocated.hpp"
#include "mlrl/common/data/tuple.hpp"
#include "mlrl/common/input/feature_vector_binary.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

#include <unordered_map>

/**
 * An abstract base class for all decorators that provide access to the values and indices of training examples stored
 * in an `AllocatedNominalFeatureVector`.
 */
typedef AbstractFeatureVectorDecorator<AllocatedNominalFeatureVector> AbstractNominalFeatureVectorDecorator;

/**
 * Provides random read and write access, as well as read and write access via iterators, to the values and indicies of
 * training examples stored in an `BinaryFeatureVector`.
 */
class BinaryFeatureVectorDecorator final : public AbstractNominalFeatureVectorDecorator {
    public:

        /**
         * @param firstView   A reference to an object of type `AllocatedNominalFeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        BinaryFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                     AllocatedMissingFeatureVector&& secondView)
            : AbstractNominalFeatureVectorDecorator(std::move(firstView), std::move(secondView)) {}

        /**
         * @param other A reference to an object of type `BinaryFeatureVectorDecorator` that should be copied
         */
        BinaryFeatureVectorDecorator(const BinaryFeatureVectorDecorator& other)
            : AbstractNominalFeatureVectorDecorator(AllocatedNominalFeatureVector(other.view.firstView),
                                                    AllocatedMissingFeatureVector()) {}

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    uint32 start, uint32 end) const override {
            return std::make_unique<EqualFeatureVector>();
        }

        std::unique_ptr<IFeatureVector> createFilteredFeatureVector(std::unique_ptr<IFeatureVector>& existing,
                                                                    const CoverageMask& coverageMask) const override {
            std::unique_ptr<BinaryFeatureVectorDecorator> filteredDecoratorPtr;
            BinaryFeatureVectorDecorator* existingDecorator =
              dynamic_cast<BinaryFeatureVectorDecorator*>(existing.get());

            if (existingDecorator) {
                // Reuse the existing feature vector...
                existing.release();
                filteredDecoratorPtr = std::unique_ptr<BinaryFeatureVectorDecorator>(existingDecorator);

                // Filter the indices of examples with missing feature values...
                MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->view.secondView;

                for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend();) {
                    uint32 index = *it;
                    it++;  // Iterator must be incremented before call to `MissingFeatureVector::set` invalidates it

                    if (!coverageMask.isCovered(index)) {
                        missingFeatureVector.set(index, false);
                    }
                }
            } else {
                // Create a new feature vector...
                filteredDecoratorPtr = std::make_unique<BinaryFeatureVectorDecorator>(*this);

                // Add the indices of examples with missing feature values...
                MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->view.secondView;

                for (auto it = this->view.secondView.indices_cbegin(); it != this->view.secondView.indices_cend();
                     it++) {
                    uint32 index = *it;

                    if (coverageMask.isCovered(index)) {
                        missingFeatureVector.set(index, true);
                    }
                }
            }

            // Filter the indices of examples associated with the minority value...
            AllocatedNominalFeatureVector& featureVector = filteredDecoratorPtr->view.firstView;
            AllocatedNominalFeatureVector::index_iterator filteredIndexIterator = featureVector.indices_begin(0);
            NominalFeatureVector::index_const_iterator indexIterator = this->view.firstView.indices_cbegin(0);
            NominalFeatureVector::index_const_iterator indicesEnd = this->view.firstView.indices_cend(0);
            uint32 maxIndices = indicesEnd - indexIterator;
            uint32 n = 0;

            for (uint32 i = 0; i < maxIndices; i++) {
                uint32 index = indexIterator[i];

                if (coverageMask.isCovered(index)) {
                    filteredIndexIterator[n] = index;
                    n++;
                }
            }

            if (n > 0) {
                featureVector.resize(n);
                return filteredDecoratorPtr;
            }

            return std::make_unique<EqualFeatureVector>();
        }
};

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
static inline std::unique_ptr<BinaryFeatureVectorDecorator> createBinaryFeatureVector(
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
            int32 nominalValue = (int32) value;

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
