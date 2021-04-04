/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/label_matrix.hpp"
#include "seco/data/matrix_dense_weights.hpp"


namespace seco {

    /**
     * An one-dimensional vector that stores a fixed number of confusion matrices in a C-contiguous array.
     */
    class DenseConfusionMatrixVector {

        private:

            float64* array_;

            uint32 numElements_;

        public:

            /**
             * @param numElements The number of elements in the vector
             */
            DenseConfusionMatrixVector(uint32 numElements);

            /**
             * @param numElements   The number of elements in the vector
             * @param init          True, if the elements of all confusion matrices should be value-initialized
             */
            DenseConfusionMatrixVector(uint32 numElements, bool init);

            /**
             * @param other A reference to an object of type `DenseConfusionMatrixVector` to be copied
             */
            DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other);

            ~DenseConfusionMatrixVector();

            /**
             * An iterator that provides access to the elements in a confusion matrix and allows to modify them.
             */
            typedef float64* iterator;

            /**
             * An iterator that provides read-only access to the elements in a confusion matrix.
             */
            typedef const float64* const_iterator;

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
             * Returns an `iterator` to the beginning of the confusion matrix at a specific position.
             *
             * @param pos   The position
             * @return      An `iterator` to the beginning
             */
            iterator confusion_matrix_begin(uint32 pos);

            /**
             * Returns an `iterator` to the end of the confusion matrix at a specific position.
             *
             * @param pos   The position
             * @return      An `iterator` to the end
             */
            iterator confusion_matrix_end(uint32 pos);

            /**
             * Returns a `const_iterator` to the beginning of the confusion matrix at a specific position.
             *
             * @param pos   The position
             * @return      A `const_iterator` to the beginning
             */
            const_iterator confusion_matrix_cbegin(uint32 pos) const;

            /**
             * Returns a `const_iterator` to the end of the confusion matrix at a specific position.
             *
             * @param pos   The position
             * @return      A `const_iterator` to the end
             */
            const_iterator confusion_matrix_cend(uint32 pos) const;

            /**
             * Returns the number of elements in the vector.
             *
             * @return The number of elements
             */
            uint32 getNumElements() const;

            /**
             * Sets the elements of all confusion matrices to zero.
             */
            void setAllToZero();

            // TODO Comment
            void add(const_iterator begin, const_iterator end);

            // TODO Comment
            void add(uint32 row, const IRandomAccessLabelMatrix& labelMatrix,
                     const DenseVector<uint8>& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                     float64 weight);

            // TODO Comment
            void addToSubset(uint32 row, const IRandomAccessLabelMatrix& labelMatrix,
                             const DenseVector<uint8>& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                             FullIndexVector::const_iterator indicesBegin, FullIndexVector::const_iterator indicesEnd,
                             float64 weight);

            // TODO Comment
            void addToSubset(uint32 row, const IRandomAccessLabelMatrix& labelMatrix,
                             const DenseVector<uint8>& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                             PartialIndexVector::const_iterator indicesBegin,
                             PartialIndexVector::const_iterator indicesEnd, float64 weight);

    };

}
