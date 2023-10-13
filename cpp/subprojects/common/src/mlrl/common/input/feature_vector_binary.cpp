#include "mlrl/common/input/feature_vector_binary.hpp"

#include "mlrl/common/input/feature_vector_equal.hpp"

BinaryFeatureVector::BinaryFeatureVector(uint32 numMinorityExamples, int32 minorityValue, int32 majorityValue)
    : NominalFeatureVector(1, numMinorityExamples, majorityValue) {
    this->values_begin()[0] = minorityValue;
    this->indptr_begin()[0] = 0;
}

std::unique_ptr<IFeatureVector> BinaryFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, uint32 start, uint32 end) const {
    return std::make_unique<EqualFeatureVector>();
}

std::unique_ptr<IFeatureVector> BinaryFeatureVector::createFilteredFeatureVector(
  std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) const {
    index_const_iterator indexIterator = this->indices_cbegin(0);
    index_const_iterator indicesEnd = this->indices_cend(0);
    uint32 maxIndices = indicesEnd - indexIterator;
    std::unique_ptr<BinaryFeatureVector> filteredFeatureVectorPtr;
    BinaryFeatureVector* existingPtr = dynamic_cast<BinaryFeatureVector*>(existing.get());

    if (existingPtr) {
        existing.release();
        filteredFeatureVectorPtr = std::unique_ptr<BinaryFeatureVector>(existingPtr);

        // Filter the indices of examples with missing feature values...
        for (auto it = filteredFeatureVectorPtr->missing_indices_cbegin();
             it != filteredFeatureVectorPtr->missing_indices_cend();) {
            uint32 index = *it;
            it++;

            if (!coverageMask.isCovered(index)) {
                filteredFeatureVectorPtr->setMissing(index, false);
            }
        }
    } else {
        filteredFeatureVectorPtr =
          std::make_unique<BinaryFeatureVector>(maxIndices, this->values_cbegin()[0], this->getMajorityValue());

        // Add the indices of examples with missing feature values...
        for (auto it = this->missing_indices_cbegin(); it != this->missing_indices_cend(); it++) {
            uint32 index = *it;

            if (coverageMask.isCovered(index)) {
                filteredFeatureVectorPtr->setMissing(index, true);
            }
        }
    }

    // Filter the indices of examples associated with the minority value...
    index_iterator filteredIndexIterator = filteredFeatureVectorPtr->indices_begin(0);
    uint32 n = 0;

    for (uint32 i = 0; i < maxIndices; i++) {
        uint32 index = indexIterator[i];

        if (coverageMask.isCovered(index)) {
            filteredIndexIterator[n] = index;
            n++;
        }
    }

    if (n > 0) {
        filteredFeatureVectorPtr->indices_ = (uint32*) realloc(filteredFeatureVectorPtr->indices_, n * sizeof(uint32));
        filteredFeatureVectorPtr->indptr_[1] = n;
        return filteredFeatureVectorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}
