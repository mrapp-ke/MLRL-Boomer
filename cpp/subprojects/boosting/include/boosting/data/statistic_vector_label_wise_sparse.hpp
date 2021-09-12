/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/tuple.hpp"
#include "common/data/vector_sparse_list.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include <memory>


namespace boosting {

    /**
     * An one-dimensional sparse vector that stores gradients and Hessians that have been calculated using a label-wise
     * decomposable loss function. For each element in the vector a single gradient and Hessian is stored.
     */
    class SparseLabelWiseStatisticVector final {

        private:

            SparseListVector<Tuple<float64>> vector_;

        public:

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             */
            SparseLabelWiseStatisticVector(uint32 numElements);

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            SparseLabelWiseStatisticVector(uint32 numElements, bool init);

            /**
             * An iterator that provides read-only access to the elements in the vector.
             */
            typedef SparseListVector<Tuple<float64>>::const_iterator const_iterator;

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
             * Sets all gradients and Hessians in the vector to zero.
             */
            void clear();

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param begin A `const_iterator` to the beginning of the vector
             * @param end   A `const_iterator` to the end of the vector
             */
            void add(const_iterator begin, const_iterator end);

            /**
             * Adds all gradients and Hessians in another vector to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const_iterator begin, const_iterator end, float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a
             * specific weight.
             *
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const_iterator begin, const_iterator end, const CompleteIndexVector& indices,
                             float64 weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param begin     A `const_iterator` to the beginning of the vector
             * @param end       A `const_iterator` to the end of the vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const_iterator begin, const_iterator end, const PartialIndexVector& indices,
                             float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param secondBegin  A `const_iterator` to the beginning of the second vector
             * @param secondEnd    A `const_iterator` to the end of the second vector
             */
            void difference(const_iterator firstBegin, const_iterator firstEnd, const CompleteIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param secondBegin   A `const_iterator` to the beginning of the second vector
             * @param secondEnd     A `const_iterator` to the end of the second vector
             */
            void difference(const_iterator firstBegin, const_iterator firstEnd, const PartialIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);

    };

}
