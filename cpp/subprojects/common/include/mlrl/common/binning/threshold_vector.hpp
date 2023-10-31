/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/input/missing_feature_vector.hpp"

/**
 * An one-dimensional vector that stores thresholds that may be used by conditions.
 */
class ThresholdVector final : public ResizableVectorDecorator<WritableVectorDecorator<AllocatedVector<float32>>>,
                              public MissingFeatureVector {
    private:

        uint32 sparseBinIndex_;

    public:

        /**
         * @param missingFeatureVector  A reference to an object of type `MissingFeatureVector` the missing indices
         *                              should be taken from
         * @param numElements           The number of elements in the vector
         * @param init                  True, if all elements in the vector should be value-initialized, false otherwise
         */
        ThresholdVector(MissingFeatureVector& missingFeatureVector, uint32 numElements, bool init = false);

        /**
         * Returns the index of the bin, sparse values have been assigned to.
         *
         * @return The index of the bin, sparse values have been assigned to. If there is no such bin, the returned
         *         index is equal to `getNumElements()`
         */
        uint32 getSparseBinIndex() const;

        /**
         * Sets the index of the bin, sparse values have been assigned to.
         *
         * @param sparseBinIndex The index to be set
         */
        void setSparseBinIndex(uint32 sparseBinIndex);

        void setNumElements(uint32 numElements, bool freeMemory) override;
};
