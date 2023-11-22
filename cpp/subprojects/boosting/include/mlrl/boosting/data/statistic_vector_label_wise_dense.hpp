/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/statistic_view_label_wise_dense.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * An one-dimensional vector that stores aggregated gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function in a C-contiguous array. For each element in the vector a single gradient
     * and Hessian is stored.
     */
    class DenseLabelWiseStatisticVector final
        : public ClearableVectorDecorator<IterableVectorDecorator<VectorDecorator<AllocatedVector<Tuple<float64>>>>> {
        public:

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            DenseLabelWiseStatisticVector(uint32 numElements, bool init = false);

            /**
             * @param other A reference to an object of type `DenseLabelWiseStatisticVector` to be copied
             */
            DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& other);

            /**
             * Adds all gradients and Hessians in another vector to this vector.
             *
             * @param vector A reference to an object of type `DenseLabelWiseStatisticVector` that stores the gradients
             *               and Hessians to be added to this vector
             */
            void add(const DenseLabelWiseStatisticVector& vector);

            /**
             * Adds all gradients and Hessians in a single row of a `DenseLabelWiseStatisticConstView` to this vector.
             *
             * @param view  A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *              gradients and Hessians to be added to this vector
             * @param row   The index of the row to be added to this vector
             */
            void add(const DenseLabelWiseStatisticConstView& view, uint32 row);

            /**
             * Adds all gradients and Hessians in a single row of a `DenseLabelWiseStatisticConstView` to this vector.
             * The gradients and Hessians to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const DenseLabelWiseStatisticConstView& view, uint32 row, float64 weight);

            /**
             * Removes all gradients and Hessians in a single row of a `DenseLabelWiseStatisticConstView` from this
             * vector.
             *
             * @param view  A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *              gradients and Hessians to be removed from this vector
             * @param row   The index of the row to be removed from this vector
             */
            void remove(const DenseLabelWiseStatisticConstView& view, uint32 row);

            /**
             * Removes all gradients and Hessians in a single row of a `DenseLabelWiseStatisticConstView` from this
             * vector. The gradients and Hessians to be removed are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be removed from this vector
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void remove(const DenseLabelWiseStatisticConstView& view, uint32 row, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `DenseLabelWiseStatisticConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             */
            void addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                             const CompleteIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in single row of a `DenseLabelWiseStatisticConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector.
             *
             * @param view      A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             */
            void addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                             const PartialIndexVector& indices);

            /**
             * Adds certain gradients and Hessians in a single row of a `DenseLabelWiseStatisticConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in single row of a `DenseLabelWiseStatisticConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector. The gradients and Hessians to be added are
             * multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseLabelWiseStatisticConstView` that stores the
             *                  gradients and Hessians to be added to this vector
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const DenseLabelWiseStatisticConstView& view, uint32 row,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `DenseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const DenseLabelWiseStatisticVector& first, const CompleteIndexVector& firstIndices,
                            const DenseLabelWiseStatisticVector& second);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `DenseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const DenseLabelWiseStatisticVector& first, const PartialIndexVector& firstIndices,
                            const DenseLabelWiseStatisticVector& second);
    };

}
