/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include "common/input/missing_feature_vector.hpp"


/**
 * An one-dimensional vector that stores thresholds that may be used by conditions.
 */
class ThresholdVector final : public MissingFeatureVector {

    private:

        DenseVector<float32> vector_;

    public:

        /**
         * @param missingFeatureVector  A reference to an object of type `MissingFeatureVector` the missing indices
         *                              should be taken from
         * @param numElements           The number of elements in the vector
         */
        ThresholdVector(MissingFeatureVector& missingFeatureVector, uint32 numElements);

        /**
         * @param missingFeatureVector  A reference to an object of type `MissingFeatureVector` the missing indices
         *                              should be taken from
         * @param numElements           The number of elements in the vector
         * @param init                  True, if all elements in the vector should be value-initialized, false otherwise
         */
        ThresholdVector(MissingFeatureVector& missingFeatureVector, uint32 numElements, bool init);

        typedef DenseVector<float32>::iterator iterator;

        typedef const DenseVector<float32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

};
