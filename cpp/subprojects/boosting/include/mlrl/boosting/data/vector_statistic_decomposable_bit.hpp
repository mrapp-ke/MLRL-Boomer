/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_decomposable_bit.hpp"
#include "mlrl/common/data/view_composite.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * An one-dimensional vector that stores aggregated gradients and Hessians that have been calculated using a
     * decomposable loss function in a bit vector. For each element in the vector a single gradient and Hessian is
     * stored.
     */
    class BitDecomposableStatisticVector final
        : public CompositeView<AllocatedBitVector<uint32>, AllocatedBitVector<uint32>> {
        public:

            /**
             * @param numElements       The number of gradients and Hessians in the vector
             * @param numBitsPerElement The number of bits per element in the bit vector
             * @param init              True, if all gradients and Hessians in the vector should be initialized with
             *                          zero, false otherwise
             */
            BitDecomposableStatisticVector(uint32 numElements, uint32 numBitsPerElement, bool init = false);

            /**
             * @param other A reference to an object of type `BitDecomposableStatisticVector` to be copied
             */
            BitDecomposableStatisticVector(const BitDecomposableStatisticVector& other);

            /**
             * Returns the number of gradients and Hessians in the vector.
             *
             * @return The number of gradients and Hessians
             */
            uint32 getNumElements() const;

            /**
             * Returns the number of bits per gradient or Hessian in the vector.
             *
             * @return The number of bits per gradient or Hessian
             */
            uint32 getNumBitsPerElement() const;

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param vector A reference to an object of type `BitDecomposableStatisticVector` that stores the gradients
             *               and Hessians to be added to this vector
             */
            void add(const BitDecomposableStatisticVector& vector);

            /**
             * Adds all gradients and Hessians in a single row of a `BitDecomposableStatisticView` to this vector.
             *
             * @param view  A reference to an object of type `BitDecomposableStatisticView` that stores the gradients
             *              and Hessians to be added to this vector
             * @param row   The index of the row to be added to this vector
             */
            void add(const BitDecomposableStatisticView& view, uint32 row);

            /**
             * Adds all gradients and Hessians in a single row of a `CContiguousView` to this vector. The gradients and
             * Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const BitDecomposableStatisticView& view, uint32 row, float64 weight);

            /**
             * Removes all gradients and Hessians in a single row of a `CContiguousView` from this vector.
             *
             * @param view  A reference to an object of type `CContiguousView` that stores the gradients and Hessians to
             *              be removed from this vector
             * @param row   The index of the row to be removed from this vector
             */
            void remove(const BitDecomposableStatisticView& view, uint32 row);

            /**
             * Removes all gradients and Hessians in a single row of a `CContiguousView` from this vector. The gradients
             * and Hessians to be removed are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be removed from this vector
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void remove(const BitDecomposableStatisticView& view, uint32 row, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `CContiguousView`, whose positions are given as
             * a `CompleteIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             */
            void addToSubset(const BitDecomposableStatisticView& view, uint32 row, const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in single row of a `CContiguousView`, whose positions are given as a
             * `PartialIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             */
            void addToSubset(const BitDecomposableStatisticView& view, uint32 row, const PartialIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `CContiguousView`, whose positions are given as
             * a `CompleteIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a
             * specific weight.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const BitDecomposableStatisticView& view, uint32 row, const CompleteIndexVector& indices,
                             float64 weight);

            /**
             * Adds certain gradients and Hessians in single row of a `CContiguousView`, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const BitDecomposableStatisticView& view, uint32 row, const PartialIndexVector& indices,
                             float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `BitDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `BitDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const BitDecomposableStatisticVector& first, const CompleteIndexVector& firstIndices,
                            const BitDecomposableStatisticVector& second);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `BitDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `BitDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const BitDecomposableStatisticVector& first, const PartialIndexVector& firstIndices,
                            const BitDecomposableStatisticVector& second);

            /**
             * Sets all gradients and Hessians stored in the vector to zero.
             */
            void clear();
    };

}
