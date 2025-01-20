/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/view_statistic_non_decomposable_dense.hpp"
#include "mlrl/boosting/iterator/iterator_diagonal.hpp"
#include "mlrl/common/data/view_vector_composite.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * An one-dimensional vector that stores gradients and Hessians that have been calculated using a non-decomposable
     * loss function in C-contiguous arrays. For each element in the vector a single gradient, but multiple Hessians are
     * stored. In a vector that stores `n` gradients `(n * (n + 1)) / 2` Hessians are stored. The Hessians can be viewed
     * as a symmetric Hessian matrix with `n` rows and columns.
     *
     * @tparam StatisticType The type of the gradients and Hessians
     */
    template<typename StatisticType>
    class DenseNonDecomposableStatisticVector final
        : public ClearableViewDecorator<
            ViewDecorator<CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>>> {
        public:

            /**
             * @param numGradients The number of gradients in the vector
             * @param init         True, if all gradients and Hessians in the vector should be initialized with zero,
             *                     false otherwise
             */
            DenseNonDecomposableStatisticVector(uint32 numGradients, bool init = false);

            /**
             * @param other A reference to an object of type `DenseNonDecomposableStatisticVector` to be copied
             */
            DenseNonDecomposableStatisticVector(const DenseNonDecomposableStatisticVector<StatisticType>& other);

            /**
             * An iterator that provides access to the gradients in the vector and allows to modify them.
             */
            typedef typename View<StatisticType>::iterator gradient_iterator;

            /**
             * An iterator that provides read-only access to the gradients in the vector.
             */
            typedef typename View<StatisticType>::const_iterator gradient_const_iterator;

            /**
             * An iterator that provides access to the Hessians in the vector and allows to modify them.
             */
            typedef typename View<StatisticType>::iterator hessian_iterator;

            /**
             * An iterator that provides read-only access to the Hessians in the vector.
             */
            typedef typename View<StatisticType>::const_iterator hessian_const_iterator;

            /**
             * An iterator that provides read-only access to the Hessians that correspond to the diagonal of the Hessian
             * matrix.
             */
            typedef DiagonalIterator<const StatisticType> hessian_diagonal_const_iterator;

            /**
             * Returns a `gradient_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_iterator` to the beginning
             */
            gradient_iterator gradients_begin();

            /**
             * Returns a `gradient_iterator` to the end of the gradients.
             *
             * @return A `gradient_iterator` to the end
             */
            gradient_iterator gradients_end();

            /**
             * Returns a `gradient_const_iterator` to the beginning of the gradients.
             *
             * @return A `gradient_const_iterator` to the beginning
             */
            gradient_const_iterator gradients_cbegin() const;

            /**
             * Returns a `gradient_const_iterator` to the end of the gradients.
             *
             * @return A `gradient_const_iterator` to the end
             */
            gradient_const_iterator gradients_cend() const;

            /**
             * Returns a `hessian_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_iterator` to the beginning
             */
            hessian_iterator hessians_begin();

            /**
             * Returns a `hessian_iterator` to the end of the Hessians.
             *
             * @return A `hessian_iterator` to the end
             */
            hessian_iterator hessians_end();

            /**
             * Returns a `hessian_const_iterator` to the beginning of the Hessians.
             *
             * @return A `hessian_const_iterator` to the beginning
             */
            hessian_const_iterator hessians_cbegin() const;

            /**
             * Returns a `hessian_const_iterator` to the end of the Hessians.
             *
             * @return A `hessian_const_iterator` to the end
             */
            hessian_const_iterator hessians_cend() const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the beginning of the Hessians that correspond to the
             * diagonal of the Hessian matrix.
             *
             * @return A `hessian_diagonal_const_iterator` to the beginning
             */
            hessian_diagonal_const_iterator hessians_diagonal_cbegin() const;

            /**
             * Returns a `hessian_diagonal_const_iterator` to the end of the Hessians that correspond to the diagonal of
             * the Hessian matrix.
             *
             * @return A `hessian_diagonal_const_iterator` to the end
             */
            hessian_diagonal_const_iterator hessians_diagonal_cend() const;

            /**
             * Returns the number of gradients in the vector.
             *
             * @return The number of gradients
             */
            uint32 getNumGradients() const;

            /**
             * Returns the number of Hessians in the vector.
             *
             + @return The number of Hessians
             */
            uint32 getNumHessians() const;

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param view A reference to an object of type `DenseNonDecomposableStatisticVector` that stores the
             *             gradients and Hessians to be added to this vector
             */
            void add(const DenseNonDecomposableStatisticVector<StatisticType>& view);

            /**
             * Adds all gradients and Hessians in a single row of a `DenseNonDecomposableStatisticView` to this vector.
             *
             * @param view  A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *              gradients and Hessians to be added to this vector
             * @param row   The index of the row to be added to this vector
             */
            void add(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row);

            /**
             * Adds all gradients and Hessians in a single row of a `DenseNonDecomposableStatisticView` to this vector.
             * The gradients and Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight);

            /**
             * Removes all gradients and Hessians in a single row of a `DenseNonDecomposableStatisticView` from this
             * vector.
             *
             * @param view  A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *              gradients and Hessians to be removed from this vector
             * @param row   The index of the row to be removed from this vector
             */
            void remove(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row);

            /**
             * Removes all gradients and Hessians in a single row of a `DenseNonDecomposableStatisticView` from this
             * vector. The gradients and Hessians to be removed are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *                  gradients and Hessians to be removed from this vector
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void remove(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row, StatisticType weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector` that provides access to the indices
             */
            void addToSubset(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row,
                             const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector` that provides access to the indices
             */
            void addToSubset(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row,
                             const PartialIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a
             * specific weight.
             *
             * @param view      A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector` that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row,
                             const CompleteIndexVector& indices, StatisticType weight);

            /**
             * Adds certain gradients and Hessians in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The gradients and Hessians to be added are multiplied by a specific
             * weight.
             *
             * @param view      A reference to an object of type `DenseNonDecomposableStatisticView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector` that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const DenseNonDecomposableStatisticView<StatisticType>& view, uint32 row,
                             const PartialIndexVector& indices, StatisticType weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `DenseNonDecomposableStatisticVector` that stores
             *                      the gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseNonDecomposableStatisticVector` that stores
             *                      the gradients and Hessians in the second vector
             */
            void difference(const DenseNonDecomposableStatisticVector<StatisticType>& first,
                            const CompleteIndexVector& firstIndices,
                            const DenseNonDecomposableStatisticVector<StatisticType>& second);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `DenseNonDecomposableStatisticVector` that stores
             *                      the gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseNonDecomposableStatisticVector` that stores
             *                      the gradients and Hessians in the second vector
             */
            void difference(const DenseNonDecomposableStatisticVector<StatisticType>& first,
                            const PartialIndexVector& firstIndices,
                            const DenseNonDecomposableStatisticVector<StatisticType>& second);
    };

}
