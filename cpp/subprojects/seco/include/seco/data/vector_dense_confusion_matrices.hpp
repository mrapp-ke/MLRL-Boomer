/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


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
             * An iterator that provides access to the elements in a confusion matrix and allows to modify them.
             */
            typedef float64* iterator;

            /**
             * An iterator that provides read-only access to the elements in a confusion matrix.
             */
            typedef const float64* const_iterator;

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

    };

}
