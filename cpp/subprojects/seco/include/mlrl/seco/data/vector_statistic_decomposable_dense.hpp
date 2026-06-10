/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/data/view_vector_composite.hpp"
#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/seco/data/view_statistic_decomposable_sparse.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

namespace seco {

    /**
     * An one-dimensional view that provides access to the number of examples for which a rule predicts individual
     * labels correctly or incorrectly, respectively.
     *
     * @tparam StatisticType The type of the counts stored in the view
     */
    template<typename StatisticType>
    class MLRLSECO_API DenseDecomposableStatisticVectorView final
        : public CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>> {
        public:

            /**
             * @param numElements   The number of elements in the view
             * @param init          True, if all elements in the view should be value-initialized, false otherwise
             */
            DenseDecomposableStatisticVectorView(uint32 numElements, bool init = false);

            /**
             * The type of the counts stored in the view.
             */
            using statistic_type = StatisticType;

            /**
             * An iterator that provides access to the counts in the view and allows to modify them.
             */
            using iterator = View<StatisticType>::iterator;

            /**
             * An iterator that provides read-only access to the counts in the view.
             */
            using const_iterator = View<StatisticType>::const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of the counts that correspond to examples for which a rule
             * predicts correctly.
             *
             * @return A `const_iterator` to the beginning of the counts that correspond to examples for which a rule
             *         predicts correctly
             */
            typename View<StatisticType>::const_iterator correct_indices_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the counts that correspond to examples for which a rule predicts
             * correctly.
             *
             * @return A `const_iterator` to the end of the counts that correspond to examples for which a rule predicts
             *         correctly
             */
            typename View<StatisticType>::const_iterator correct_indices_cend() const;

            /**
             * Returns an `iterator` to the beginning of the counts that correspond to examples for which a rule
             * predicts correctly.
             *
             * @return An `iterator` to the beginning of the counts that correspond to examples for which a rule
             *         predicts correctly
             */
            typename View<StatisticType>::iterator correct_indices_begin();

            /**
             * Returns an `iterator` to the end of the counts that correspond to examples for which a rule predicts
             * correctly.
             *
             * @return An `iterator` to the end of the counts that correspond to examples for which a rule predicts
             *         correctly
             */
            typename View<StatisticType>::iterator correct_indices_end();

            /**
             * Returns a `const_iterator` to the beginning of the counts that correspond to examples  for which a rule
             * predicts incorrectly.
             *
             * @return A `const_iterator` to the beginning of the counts that correspond to examples  for which a rule
             *         predicts incorrectly
             */
            typename View<StatisticType>::const_iterator incorrect_indices_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the counts that correspond to examples  for which a rule
             * predicts incorrectly.
             *
             * @return A `const_iterator` to the end of the counts that correspond to examples  for which a rule
             *         predicts incorrectly
             */
            typename View<StatisticType>::const_iterator incorrect_indices_cend() const;

            /**
             * Returns an `iterator` to the beginning of the counts that correspond to examples for which a rule
             * predicts incorrectly.
             *
             * @return An `iterator` to the beginning of the counts that correspond to examples for which a rule
             *         predicts incorrectly
             */
            typename View<StatisticType>::iterator incorrect_indices_begin();

            /**
             * Returns an `iterator` to the end of the counts that correspond to examples for which a rule predicts
             * incorrectly.
             *
             * @return An `iterator` to the end of the counts that correspond to examples for which a rule predicts
             *         incorrectly
             */
            typename View<StatisticType>::iterator incorrect_indices_end();

            /**
             * Returns the number of elements in the view.
             *
             * @return The number of elements
             */
            uint32 getNumElements() const;

            /**
             * Sets all counts stored in the view to zero.
             */
            void clear();
    };

    /**
     * An one-dimensional vector that stores the number of examples for which a rule predicts individual labels
     * correctly or incorrectly, respectively, in two C-contiguous arrays.
     *
     * @tparam StatisticType    The type of the counts stored in the vector
     * @tparam VectorMath       The type that implements basic operations for calculating with numerical arrays
     */
    template<typename StatisticType, typename VectorMath>
    class DenseDecomposableStatisticVector final
        : public ClearableViewDecorator<ViewDecorator<DenseDecomposableStatisticVectorView<StatisticType>>> {
        public:

            /**
             * @param numElements   The number of elements in the vector
             * @param init          True, if all elements in the vector should be value-initialized, false otherwise
             */
            DenseDecomposableStatisticVector(uint32 numElements, bool init = false);

            /**
             * @param other A reference to an object of type `DenseDecomposableStatisticVector` to be copied
             */
            DenseDecomposableStatisticVector(const DenseDecomposableStatisticVector<StatisticType, VectorMath>& other);

            /**
             * Returns a `const_iterator` to the beginning of the counts that correspond to examples for which a rule
             * predicts correctly.
             *
             * @return A `const_iterator` to the beginning of the counts that correspond to examples for which a rule
             *         predicts correctly
             */
            typename View<StatisticType>::const_iterator correct_indices_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the counts that correspond to examples for which a rule predicts
             * correctly.
             *
             * @return A `const_iterator` to the end of the counts that correspond to examples for which a rule predicts
             *         correctly
             */
            typename View<StatisticType>::const_iterator correct_indices_cend() const;

            /**
             * Returns an `iterator` to the beginning of the counts that correspond to examples for which a rule
             * predicts correctly.
             *
             * @return An `iterator` to the beginning of the counts that correspond to examples for which a rule
             *         predicts correctly
             */
            typename View<StatisticType>::iterator correct_indices_begin();

            /**
             * Returns an `iterator` to the end of the counts that correspond to examples for which a rule predicts
             * correctly.
             *
             * @return An `iterator` to the end of the counts that correspond to examples for which a rule predicts
             *         correctly
             */
            typename View<StatisticType>::iterator correct_indices_end();

            /**
             * Returns a `const_iterator` to the beginning of the counts that correspond to examples  for which a rule
             * predicts incorrectly.
             *
             * @return A `const_iterator` to the beginning of the counts that correspond to examples  for which a rule
             *         predicts incorrectly
             */
            typename View<StatisticType>::const_iterator incorrect_indices_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the counts that correspond to examples  for which a rule
             * predicts incorrectly.
             *
             * @return A `const_iterator` to the end of the counts that correspond to examples  for which a rule
             *         predicts incorrectly
             */
            typename View<StatisticType>::const_iterator incorrect_indices_cend() const;

            /**
             * Returns an `iterator` to the beginning of the counts that correspond to examples for which a rule
             * predicts incorrectly.
             *
             * @return An `iterator` to the beginning of the counts that correspond to examples for which a rule
             *         predicts incorrectly
             */
            typename View<StatisticType>::iterator incorrect_indices_begin();

            /**
             * Returns an `iterator` to the end of the counts that correspond to examples for which a rule predicts
             * incorrectly.
             *
             * @return An `iterator` to the end of the counts that correspond to examples for which a rule predicts
             *         incorrectly
             */
            typename View<StatisticType>::iterator incorrect_indices_end();

            /**
             * Returns the number of elements in the view.
             *
             * @return The number of elements
             */
            uint32 getNumElements() const;

            /**
             * Adds all counts in another vector to this vector.
             *
             * @param other A reference to an object of type `DenseDecomposableStatisticVectorView` that stores the
             *              counts to be added to this vector
             */
            void add(const DenseDecomposableStatisticVectorView<StatisticType>& other);

            /**
             * Increases the counts in this vector based on a specific row in a `SparseDecomposableStatisticView`. The
             * increments of the counts are multiplied by a given weight.
             *
             * @param view      A reference to an object of type `SparseDecomposableStatisticView`
             * @param row       The index of the row to be used for updating this vector
             * @param weight    The weight, the increments of the counts should be multiplied by
             */
            void add(const SparseDecomposableStatisticView& view, uint32 row, StatisticType weight = 1);

            /**
             * Decreases the counts in this vector based on a specific row in a `SparseDecomposableStatisticView`. The
             * decrements of the counts are multiplied by a given weight.
             *
             * @param view      A reference to an object of type `SparseDecomposableStatisticView`
             * @param row       The index of the row to be used for updating this vector
             * @param weight    The weight, the decrements of the counts should be multiplied by
             */
            void remove(const SparseDecomposableStatisticView& view, uint32 row, StatisticType weight = 1);

            /**
             * Increments the counts in this vector based on certain elements of a specific row in a
             * `SparseDecomposableStatisticView`, whose positions are given as a `CompleteIndexVector`. The increments
             * of the counts are multiplied by a given weight.
             *
             * @param view      A reference to an object of type `SparseDecomposableStatisticView`
             * @param row       The index of the row to be used for updating this vector
             * @param indices   A reference to a `CompleteIndexVector` that provides access to the indices
             * @param weight    The weight, the increments of the counts should be multiplied by
             */
            void addToSubset(const SparseDecomposableStatisticView& view, uint32 row,
                             const CompleteIndexVector& indices, StatisticType weight = 1);

            /**
             * Increments the counts in this vector based on certain elements of a specific row in a
             * `SparseDecomposableStatisticView`, whose positions are given as a `PartialIndexVector`. The increments of
             * the counts are multiplied by a given weight.
             *
             * @param view      A reference to an object of type `SparseDecomposableStatisticView`
             * @param row       The index of the row to be used for updating this vector
             * @param indices   A reference to a `PartialIndexVector` that provides access to the indices
             * @param weight    The weight, the increments of the counts should be multiplied by
             */
            void addToSubset(const SparseDecomposableStatisticView& view, uint32 row, const PartialIndexVector& indices,
                             StatisticType weight = 1);

            /**
             * Sets the counts in this vector to the difference `first - second` between the elements in two other
             * vectors, considering only the elements in the first vector that correspond to the positions provided by a
             * `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `DenseDecomposableStatisticVectorView` that stores
             *                      the elements in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseDecomposableStatisticVectorView` that stores
             *                      the elements in the second vector
             */
            void difference(const DenseDecomposableStatisticVectorView<StatisticType>& first,
                            const CompleteIndexVector& firstIndices,
                            const DenseDecomposableStatisticVectorView<StatisticType>& second);

            /**
             * Sets the counts in this vector to the difference `first - second` between the elements in two other
             * vectors, considering only the elements in the first vector that correspond to the positions provided by a
             * `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `DenseDecomposableStatisticVectorView` that stores
             *                      the elements in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseDecomposableStatisticVectorView` that stores
             *                      the elements in the second vector
             */
            void difference(const DenseDecomposableStatisticVectorView<StatisticType>& first,
                            const PartialIndexVector& firstIndices,
                            const DenseDecomposableStatisticVectorView<StatisticType>& second);
    };

}
