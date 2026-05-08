/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/data/view_vector_composite.hpp"
#include "mlrl/seco/data/matrix_statistic_decomposable_dense.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

namespace seco {

    /**
     * An one-dimensional view that provides access to a fixed number of confusion matrices in a pre-allocated
     * C-contiguous array.
     *
     * @tparam StatisticType The type of the elements stored in the confusion matrices
     */
    template<typename StatisticType>
    class MLRLSECO_API DenseConfusionMatrixVectorView final
        : public CompositeVector<CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>,
                                 CompositeVector<AllocatedVector<StatisticType>, AllocatedVector<StatisticType>>> {
        public:

            /**
             * @param numElements   The number of elements in the view
             * @param init          True, if all elements in the view should be value-initialized, false otherwise
             */
            DenseConfusionMatrixVectorView(uint32 numElements, bool init = false);

            /**
             * The type of the confusion matrix elements.
             */
            using statistic_type = StatisticType;

            /**
             * An iterator that provides access to confusion matrix elements in the view and allows to modify them.
             */
            using iterator = View<StatisticType>::iterator;

            /**
             * An iterator that provides read-only access to confusion matrix elements in the view.
             */
            using const_iterator = View<StatisticType>::const_iterator;

            /**
             * Returns a `const_iterator` to the beginning of the irrelevant labels for which a rule predicts
             * negatively.
             *
             * @return A `const_iterator` to the beginning of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator in_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the irrelevant labels for which a rule predicts negatively.
             *
             * @return A `const_iterator` to the end of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator in_cend() const;

            /**
             * Returns an `iterator` to the beginning of the irrelevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the beginning of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator in_begin();

            /**
             * Returns an `iterator` to the end of the irrelevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the end of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator in_end();

            /**
             * Returns a `const_iterator` to the beginning of the irrelevant labels for which a rule predicts
             * positively.
             *
             * @return A `const_iterator` to the beginning of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator ip_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the irrelevant labels for which a rule predicts positively.
             *
             * @return A `const_iterator` to the end of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator ip_cend() const;

            /**
             * Returns an `iterator` to the beginning of the irrelevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the beginning of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator ip_begin();

            /**
             * Returns an `iterator` to the end of the irrelevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the end of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator ip_end();

            /**
             * Returns a `const_iterator` to the beginning of the relevant labels for which a rule predicts negatively.
             *
             * @return A `const_iterator` to the beginning of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator rn_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the relevant labels for which a rule predicts negatively.
             *
             * @return A `const_iterator` to the end of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator rn_cend() const;

            /**
             * Returns an `iterator` to the beginning of the relevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the beginning of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator rn_begin();

            /**
             * Returns an `iterator` to the end of the relevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the end of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator rn_end();

            /**
             * Returns a `const_iterator` to the beginning of the relevant labels for which a rule predicts positively.
             *
             * @return A `const_iterator` to the beginning of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator rp_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the relevant labels for which a rule predicts positively.
             *
             * @return A `const_iterator` to the end of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator rp_cend() const;

            /**
             * Returns an `iterator` to the beginning of the relevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the beginning of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator rp_begin();

            /**
             * Returns an `iterator` to the end of the relevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the end of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator rp_end();

            /**
             * Returns the number of elements in the view.
             *
             * @return The number of elements
             */
            uint32 getNumElements() const;

            /**
             * Sets all values stored in the view to zero.
             */
            void clear();
    };

    /**
     * An one-dimensional vector that stores a fixed number of confusion matrices in a C-contiguous array.
     *
     * @tparam StatisticType    The type of the elements stored in the confusion matrices
     * @tparam VectorMath       The type that implements basic operations for calculating with numerical arrays
     */
    template<typename StatisticType, typename VectorMath>
    class DenseConfusionMatrixVector final
        : public ClearableViewDecorator<ViewDecorator<DenseConfusionMatrixVectorView<StatisticType>>> {
        public:

            /**
             * @param numElements   The number of elements in the vector
             * @param init          True, if the elements of all confusion matrices should be value-initialized
             */
            DenseConfusionMatrixVector(uint32 numElements, bool init = false);

            /**
             * @param other A reference to an object of type `DenseConfusionMatrixVector` to be copied
             */
            DenseConfusionMatrixVector(const DenseConfusionMatrixVector<StatisticType, VectorMath>& other);

            /**
             * Returns a `const_iterator` to the beginning of the irrelevant labels for which a rule predicts
             * negatively.
             *
             * @return A `const_iterator` to the beginning of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator in_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the irrelevant labels for which a rule predicts negatively.
             *
             * @return A `const_iterator` to the end of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator in_cend() const;

            /**
             * Returns an `iterator` to the beginning of the irrelevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the beginning of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator in_begin();

            /**
             * Returns an `iterator` to the end of the irrelevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the end of the irrelevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator in_end();

            /**
             * Returns a `const_iterator` to the beginning of the irrelevant labels for which a rule predicts
             * positively.
             *
             * @return A `const_iterator` to the beginning of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator ip_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the irrelevant labels for which a rule predicts positively.
             *
             * @return A `const_iterator` to the end of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator ip_cend() const;

            /**
             * Returns an `iterator` to the beginning of the irrelevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the beginning of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator ip_begin();

            /**
             * Returns an `iterator` to the end of the irrelevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the end of the irrelevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator ip_end();

            /**
             * Returns a `const_iterator` to the beginning of the relevant labels for which a rule predicts negatively.
             *
             * @return A `const_iterator` to the beginning of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator rn_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the relevant labels for which a rule predicts negatively.
             *
             * @return A `const_iterator` to the end of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::const_iterator rn_cend() const;

            /**
             * Returns an `iterator` to the beginning of the relevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the beginning of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator rn_begin();

            /**
             * Returns an `iterator` to the end of the relevant labels for which a rule predicts negatively.
             *
             * @return An `iterator` to the end of the relevant labels for which a rule predicts negatively
             */
            typename View<StatisticType>::iterator rn_end();

            /**
             * Returns a `const_iterator` to the beginning of the relevant labels for which a rule predicts positively.
             *
             * @return A `const_iterator` to the beginning of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator rp_cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the relevant labels for which a rule predicts positively.
             *
             * @return A `const_iterator` to the end of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::const_iterator rp_cend() const;

            /**
             * Returns an `iterator` to the beginning of the relevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the beginning of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator rp_begin();

            /**
             * Returns an `iterator` to the end of the relevant labels for which a rule predicts positively.
             *
             * @return An `iterator` to the end of the relevant labels for which a rule predicts positively
             */
            typename View<StatisticType>::iterator rp_end();

            /**
             * Returns the number of elements in the view.
             *
             * @return The number of elements
             */
            uint32 getNumElements() const;

            /**
             * Adds all confusion matrix elements in another vector to this vector.
             *
             * @param other A reference to an object of type `DenseConfusionMatrixVectorView`  that stores the confusion
             *              matrices to be added to this vector
             */
            void add(const DenseConfusionMatrixVectorView<StatisticType>& other);

            /**
             * Adds the confusion matrix elements that correspond to an example at a specific index to this vector. The
             * confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type
             *                  `DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View` that provides
             *                  random access to the confusion matrices of the training examples
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void add(const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
                     StatisticType weight = 1);

            /**
             * Adds the confusion matrix elements that correspond to an example at a specific index to this vector. The
             * confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseDecomposableStatisticMatrix<BinaryCsrView>::View`
             *                  that provides random access to the confusion matrices of the training examples
             * @param row       The index of the row to be added to this vector
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void add(const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row,
                     StatisticType weight = 1);

            /**
             * Removes the confusion matrix elements at a specific row of a view from this vector.  The confusion matrix
             * elements to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type
             *                  `DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View` that provides
             *                  random access to the confusion matrices of the training examples
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void remove(const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view, uint32 row,
                        StatisticType weight = 1);

            /**
             * Removes the confusion matrix elements at a specific row of a view from this vector. The confusion matrix
             * elements to be added are multiplied by a specific weight.
             *
             * @param view      A reference to an object of type `DenseDecomposableStatisticMatrix<BinaryCsrView>::View`
             *                  that provides random access to the confusion matrices of the training examples
             * @param row       The index of the row to be removed from this vector
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void remove(const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row,
                        StatisticType weight = 1);

            /**
             * Adds certain confusion matrix elements in a specific row of a view, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param view      A reference to an object of type
             *                  `DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View` that provides
             *                  random access to the confusion matrices of the training examples
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector` that provides access to the indices
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view,
                             uint32 row, const CompleteIndexVector& indices, StatisticType weight = 1);

            /**
             * Adds certain confusion matrix elements in a specific row of a view, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param view      A reference to an object of type `DenseDecomposableStatisticMatrix<BinaryCsrView>::View`
             *                  that provides row-wise access to the labels of the training examples
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `CompleteIndexVector` that provides access to the indices
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row,
                             const CompleteIndexVector& indices, StatisticType weight = 1);

            /**
             * Adds certain confusion matrix elements in a specific row of a view, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param view      A reference to an object of type
             *                  `DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View` that provides
             *                  random access to the confusion matrices of the training examples
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector` that provides access to the indices
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(const DenseDecomposableStatisticMatrix<CContiguousView<const uint8>>::View& view,
                             uint32 row, const PartialIndexVector& indices, StatisticType weight = 1);

            /**
             * Adds certain confusion matrix elements in a specific row of a view, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param view      A reference to an object of type `DenseDecomposableStatisticMatrix<BinaryCsrView>::View`
             *                  that provides row-wise access to the labels of the training examples
             * @param row       The index of the row to be added to this vector
             * @param indices   A reference to a `PartialIndexVector` that provides access to the indices
             * @param weight    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(const DenseDecomposableStatisticMatrix<BinaryCsrView>::View& view, uint32 row,
                             const PartialIndexVector& indices, StatisticType weight = 1);

            /**
             * Sets the confusion matrix elements in this vector to the difference `first - second` between the elements
             * in two other vectors, considering only the elements in the first vector that correspond to the positions
             * provided by a `CompleteIndexVector`.
             *
             * @param first         A reference to an object of type `DenseConfusionMatrixVectorView` that stores the
             *                      confusion matrices in the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseConfusionMatrixVectorView` that stores the
             *                      confusion matrices in the second vector
             */
            void difference(const DenseConfusionMatrixVectorView<StatisticType>& first,
                            const CompleteIndexVector& firstIndices,
                            const DenseConfusionMatrixVectorView<StatisticType>& second);

            /**
             * Sets the confusion matrix elements in this vector to the difference `first - second` between the elements
             * in two other vectors, considering only the elements in the first vector that correspond to the positions
             * provided by a `PartialIndexVector`.
             *
             * @param first         A reference to an object of type `DenseConfusionMatrixVectorView` that stores the
             *                      confusion matrices in the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param second        A reference to an object of type `DenseConfusionMatrixVectorView` that stores the
             *                      confusion matrices in the second vector
             */
            void difference(const DenseConfusionMatrixVectorView<StatisticType>& first,
                            const PartialIndexVector& firstIndices,
                            const DenseConfusionMatrixVectorView<StatisticType>& second);
    };

}
