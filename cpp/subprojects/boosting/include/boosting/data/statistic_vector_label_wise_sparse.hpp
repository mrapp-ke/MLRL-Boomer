/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/triple.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "boosting/data/histogram_view_label_wise_sparse.hpp"
#include "boosting/data/statistic_view_label_wise_sparse.hpp"


namespace boosting {

    /**
     * An one-dimensional vector that stores aggregated gradients and Hessians that have been calculated using a
     * label-wise decomposable loss function in a C-contiguous array. For each element in the vector a single gradient
     * and Hessian, as well as the sums of the weights of the aggregated gradients and Hessians, is stored.
     */
    class SparseLabelWiseStatisticVector final {

        private:

            uint32 numElements_;

            Triple<float64>* statistics_;

            float64 sumOfWeights_;

        public:

            /**
             * @param numElements The number of gradients and Hessians in the vector
             */
            SparseLabelWiseStatisticVector(uint32 numElements);

            /**
             * @param numElements   The number of gradients and Hessians in the vector
             * @param init          True, if all gradients and Hessians in the vector should be initialized with zero,
             *                      false otherwise
             */
            SparseLabelWiseStatisticVector(uint32 numElements, bool init);

            /**
             * @param vector A reference to an object of type `SparseLabelWiseStatisticVector` to be copied
             */
            SparseLabelWiseStatisticVector(const SparseLabelWiseStatisticVector& vector);

            ~SparseLabelWiseStatisticVector();

            /**
             * An iterator that provides read-only access to the elements in the vector.
             */
            typedef Triple<float64>* const_iterator;

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
             * @param vector A reference to an object of type `SparseLabelWiseStatisticVector` that stores the gradients
             *               and Hessians to be added to this vector
             */
            void add(const SparseLabelWiseStatisticVector& vector);

            /**
             * Adds all gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView` to this vector.
             *
             * @param begin A `SparseLabelWiseStatisticConstView::const_iterator` to the beginning of the row
             * @param end   A `SparseLabelWiseStatisticConstView::const_iterator` to the end of the row
             */
            void add(const SparseLabelWiseStatisticConstView& view, uint32 row);

            /**
             * Adds all gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView` to this vector.
             * The gradients and Hessians to be added are multiplied by a specific weight.
             *
             * @param begin     A `SparseLabelWiseStatisticConstView::const_iterator` to the beginning of the row
             * @param end       A `SparseLabelWiseStatisticConstView::const_iterator` to the end of the row
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void add(const SparseLabelWiseStatisticConstView& view, uint32 row, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param begin     A `SparseLabelWiseStatisticConstView::const_iterator` to the beginning of the row
             * @param end       A `SparseLabelWiseStatisticConstView::const_iterator` to the end of the row
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseStatisticConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector. The gradients and Hessians to be added are
             * multiplied by a specific weight.
             *
             * @param begin     A `SparseLabelWiseStatisticConstView::const_iterator` to the beginning of the row
             * @param end       A `SparseLabelWiseStatisticConstView::const_iterator` to the end of the row
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseStatisticConstView& view, uint32 row,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseHistogramConstView`, whose
             * positions are given as a `CompleteIndexVector`, to this vector. The gradients and Hessians to be added
             * are multiplied by a specific weight.
             *
             * @param begin     A `SparseLabelWiseHistogramConstView::const_iterator` to the beginning of the row
             * @param end       A `SparseLabelWiseHistogramConstView::const_iterator` to the end of the row
             * @param indices   A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain gradients and Hessians in a single row of a `SparseLabelWiseHistogramConstView`, whose
             * positions are given as a `PartialIndexVector`, to this vector. The gradients and Hessians to be added are
             * multiplied by a specific weight.
             *
             * @param begin     A `SparseLabelWiseHistogramConstView::const_iterator` to the beginning of the row
             * @param end       A `SparseLabelWiseHistogramConstView::const_iterator` to the end of the row
             * @param indices   A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight    The weight, the gradients and Hessians should be multiplied by
             */
            void addToSubset(const SparseLabelWiseHistogramConstView& view, uint32 row,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const SparseLabelWiseStatisticVector& first, const CompleteIndexVector& firstIndices,
                            const SparseLabelWiseStatisticVector& second);

            /**
             * Sets the gradients and Hessians in this vector to the difference `first - second` between the gradients
             * and Hessians in two other vectors, considering only the gradients and Hessians in the first vector that
             * correspond to the positions provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `SparseLabelWiseStatisticVector` that stores the
             *                      gradients and Hessians in the second vector
             */
            void difference(const SparseLabelWiseStatisticVector& first, const PartialIndexVector& firstIndices,
                            const SparseLabelWiseStatisticVector& second);

    };

}
