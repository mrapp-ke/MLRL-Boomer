/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"
#include "mlrl/common/data/view_matrix_c_contiguous.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/seco/data/confusion_matrix.hpp"
#include "mlrl/seco/data/matrix_coverage_dense.hpp"

namespace seco {

    /**
     * An one-dimensional vector that stores a fixed number of confusion matrices in a C-contiguous array.
     */
    class DenseConfusionMatrixVector final
        : public ClearableViewDecorator<DenseVectorDecorator<AllocatedVector<ConfusionMatrix<uint32>>>> {
        public:

            /**
             * @param numElements   The number of elements in the vector
             * @param init          True, if the elements of all confusion matrices should be value-initialized
             */
            DenseConfusionMatrixVector(uint32 numElements, bool init = false);

            /**
             * @param other A reference to an object of type `DenseConfusionMatrixVector` to be copied
             */
            DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other);

            /**
             * Adds all confusion matrix elements in another vector to this vector.
             *
             * @param begin A `const_iterator` to the beginning of the other vector
             * @param end   A `const_iterator` to the end of the other vector
             */
            void add(const_iterator begin, const_iterator end);

            /**
             * Adds the confusion matrix elements that correspond to an example at a specific index to this vector. The
             * confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `CContiguousView` that provides random
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void add(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                     View<uint32>::const_iterator majorityLabelIndicesBegin,
                     View<uint32>::const_iterator majorityLabelIndicesEnd, const DenseCoverageMatrix& coverageMatrix,
                     uint32 weight);

            /**
             * Adds the confusion matrix elements that correspond to an example at a specific index to this vector. The
             * confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `BinaryCsrView` that provides row-wise
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void add(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                     View<uint32>::const_iterator majorityLabelIndicesBegin,
                     View<uint32>::const_iterator majorityLabelIndicesEnd, const DenseCoverageMatrix& coverageMatrix,
                     uint32 weight);

            /**
             * Removes the confusion matrix elements that correspond to an example at a specific index from this vector.
             * The confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `CContiguousView` that provides random
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void remove(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                        View<uint32>::const_iterator majorityLabelIndicesBegin,
                        View<uint32>::const_iterator majorityLabelIndicesEnd, const DenseCoverageMatrix& coverageMatrix,
                        uint32 weight);

            /**
             * Removes the confusion matrix elements that correspond to an example at a specific index from this vector.
             * The confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `BinaryCsrView` that provides row-wise
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void remove(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                        View<uint32>::const_iterator majorityLabelIndicesBegin,
                        View<uint32>::const_iterator majorityLabelIndicesEnd, const DenseCoverageMatrix& coverageMatrix,
                        uint32 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `CContiguousView` that provides random
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param indices                   A reference to a `CompleteIndexVector' that provides access to the
             *                                  indices
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             View<uint32>::const_iterator majorityLabelIndicesBegin,
                             View<uint32>::const_iterator majorityLabelIndicesEnd,
                             const DenseCoverageMatrix& coverageMatrix, const CompleteIndexVector& indices,
                             uint32 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `BinaryCsrView` that provides row-wise
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param indices                   A reference to a `CompleteIndexVector' that provides access to the
             *                                  indices
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             View<uint32>::const_iterator majorityLabelIndicesBegin,
                             View<uint32>::const_iterator majorityLabelIndicesEnd,
                             const DenseCoverageMatrix& coverageMatrix, const CompleteIndexVector& indices,
                             uint32 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `CContiguousView` that provides random
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param indices                   A reference to a `PartialIndexVector' that provides access to the
             *                                  indices
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const CContiguousView<const uint8>& labelMatrix,
                             View<uint32>::const_iterator majorityLabelIndicesBegin,
                             View<uint32>::const_iterator majorityLabelIndicesEnd,
                             const DenseCoverageMatrix& coverageMatrix, const PartialIndexVector& indices,
                             uint32 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex              The index of the example
             * @param labelMatrix               A reference to an object of type `BinaryCsrView` that provides row-wise
             *                                  access to the labels of the training examples
             * @param majorityLabelIndicesBegin An iterator to the beginning of the indices of the labels that are
             *                                  relevant to the majority of the training examples
             * @param majorityLabelIndicesEnd   An iterator to the end of the indices of the labels that are relevant to
             *                                  to the majority of the training examples
             * @param coverageMatrix            A reference to an object of type `DenseCoverageMatrix` that stores how
             *                                  often individual examples and labels have been covered
             * @param indices                   A reference to a `PartialIndexVector' that provides access to the
             *                                  indices
             * @param weight                    The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const BinaryCsrView& labelMatrix,
                             View<uint32>::const_iterator majorityLabelIndicesBegin,
                             View<uint32>::const_iterator majorityLabelIndicesEnd,
                             const DenseCoverageMatrix& coverageMatrix, const PartialIndexVector& indices,
                             uint32 weight);

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
            void difference(const_iterator firstBegin, const_iterator firstEnd, const CompleteIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);

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
            void difference(const_iterator firstBegin, const_iterator firstEnd, const PartialIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);
    };

}
