/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/statistic.hpp"
#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * An one-dimensional vector that stores aggregated gradients and Hessians that have been calculated using a
     * decomposable loss function in a C-contiguous array. For each element in the vector a single gradient and Hessian
     * is stored.
     *
     * @tparam StatisticType The type of the gradient and Hessians
     */
    template<typename StatisticType>
    class DenseDecomposableStatisticVector final
        : public ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<Statistic<StatisticType>>>> {
        public:

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            DenseDecomposableStatisticVector(uint32 numElements, bool init = false);

            /**
             * @param other A reference to an object of type `DenseDecomposableStatisticVector` to be copied
             */
            DenseDecomposableStatisticVector(const DenseDecomposableStatisticVector<StatisticType>& other);

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param vector A reference to an object of type `DenseDecomposableStatisticVector` that stores the
             *               gradients and Hessians to be added to this vector
             */
            void add(const DenseDecomposableStatisticVector<StatisticType>& vector);

            /**
             * Adds all gradients and Hessians in a single row of a `CContiguousView` to this vector.
             *
             * @param view  A reference to an object of type `CContiguousView` that stores the gradients and Hessians to
             *              be added to this vector
             * @param row   The index of the row to be added to this vector
             */
            void add(const CContiguousView<Statistic<StatisticType>>& view, uint32 row);

            /**
             * Adds all gradients and Hessians in a single row of a `CContiguousView` to this vector. The gradients and
             * Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const CContiguousView<Statistic<StatisticType>>& view, uint32 row, StatisticType weight);

            /**
             * Removes all gradients and Hessians in a single row of a `CContiguousView` from this vector.
             *
             * @param view  A reference to an object of type `CContiguousView` that stores the gradients and Hessians to
             *              be removed from this vector
             * @param row   The index of the row to be removed from this vector
             */
            void remove(const CContiguousView<Statistic<StatisticType>>& view, uint32 row);

            /**
             * Removes all gradients and Hessians in a single row of a `CContiguousView` from this vector. The gradients
             * and Hessians to be removed are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be removed from this vector
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void remove(const CContiguousView<Statistic<StatisticType>>& view, uint32 row, StatisticType weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `CContiguousView`, whose positions are given as
             * a `CompleteIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             */
            void addToSubset(const CContiguousView<Statistic<StatisticType>>& view, uint32 row,
                             const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in single row of a `CContiguousView`, whose positions are given as a
             * `PartialIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `CContiguousView` that stores the gradients and
             *                  Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             */
            void addToSubset(const CContiguousView<Statistic<StatisticType>>& view, uint32 row,
                             const PartialIndexVector& indices);

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
            void addToSubset(const CContiguousView<Statistic<StatisticType>>& view, uint32 row,
                             const CompleteIndexVector& indices, StatisticType weight);

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
            void addToSubset(const CContiguousView<Statistic<StatisticType>>& view, uint32 row,
                             const PartialIndexVector& indices, StatisticType weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `DenseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const DenseDecomposableStatisticVector<StatisticType>& first,
                            const CompleteIndexVector& firstIndices,
                            const DenseDecomposableStatisticVector<StatisticType>& second);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `DenseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseDecomposableStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const DenseDecomposableStatisticVector<StatisticType>& first,
                            const PartialIndexVector& firstIndices,
                            const DenseDecomposableStatisticVector<StatisticType>& second);
    };

}
