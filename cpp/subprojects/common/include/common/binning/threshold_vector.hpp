/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dense.hpp"


/**
 * An one-dimensional vector that stores thresholds that may be used by conditions.
 */
class ThresholdVector final : public DenseVector<float32> {

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        ThresholdVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        ThresholdVector(uint32 numElements, bool init);

};
