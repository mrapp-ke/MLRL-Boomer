/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/seco/data/confusion_matrix.hpp"
#include "mlrl/seco/data/matrix_statistic_decomposable_dense.hpp"

namespace seco {

    /**
     * An one-dimensional view that provides access to a fixed number of confusion matrices in a pre-allocated
     * C-contiguous array.
     *
     * @tparam StatisticType The type of the elements stored in the confusion matrices
     */
    template<typename StatisticType>
    using DenseConfusionMatrixVectorView = AllocatedVector<ConfusionMatrix<StatisticType>>;

    /**
     * An one-dimensional vector that stores a fixed number of confusion matrices in a C-contiguous array.
     *
     * @tparam StatisticType The type of the elements stored in the confusion matrices
     */
    template<typename StatisticType>
    class DenseConfusionMatrixVector final
        : public ClearableViewDecorator<DenseVectorDecorator<DenseConfusionMatrixVectorView<StatisticType>>> {
        public:

            /**
             * @param numElements   The number of elements in the vector
             * @param init          True, if the elements of all confusion matrices should be value-initialized
             */
            DenseConfusionMatrixVector(uint32 numElements, bool init = false);

            /**
             * @param other A reference to an object of type `DenseConfusionMatrixVector` to be copied
             */
            DenseConfusionMatrixVector(const DenseConfusionMatrixVector<StatisticType>& other);

            /**
             * Adds all confusion matrix elements in another vector to this vector.
             *
             * @param other A reference to an object of type `DenseConfusionMatrixVector` to be copied
             */
            void add(const DenseConfusionMatrixVector<StatisticType>& other);

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
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param secondBegin  A `const_iterator` to the beginning of the second vector
             * @param secondEnd    A `const_iterator` to the end of the second vector
             */
            void difference(typename View<ConfusionMatrix<StatisticType>>::const_iterator firstBegin,
                            typename View<ConfusionMatrix<StatisticType>>::const_iterator firstEnd,
                            const CompleteIndexVector& firstIndices,
                            typename View<ConfusionMatrix<StatisticType>>::const_iterator secondBegin,
                            typename View<ConfusionMatrix<StatisticType>>::const_iterator secondEnd);

            /**
             * Sets the confusion matrix elements in this vector to the difference `first - second` between the elements
             * in two other vectors, considering only the elements in the first vector that correspond to the positions
             * provided by a `PartialIndexVector`.
             *
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param secondBegin   A `const_iterator` to the beginning of the second vector
             * @param secondEnd     A `const_iterator` to the end of the second vector
             */
            void difference(typename View<ConfusionMatrix<StatisticType>>::const_iterator firstBegin,
                            typename View<ConfusionMatrix<StatisticType>>::const_iterator firstEnd,
                            const PartialIndexVector& firstIndices,
                            typename View<ConfusionMatrix<StatisticType>>::const_iterator secondBegin,
                            typename View<ConfusionMatrix<StatisticType>>::const_iterator secondEnd);
    };

}
