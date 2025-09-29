/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_sparse_array_binary.hpp"
#include "mlrl/seco/util/dll_exports.hpp"

#include <memory>

namespace seco {

    /**
     * Implements row-wise read and write access to confusion matrices that are stored in pre-allocated C-contiguous
     * arrays.
     *
     * @tparam LabelMatrix      The type of the matrix that provides access to the labels of the training examples
     * @tparam CoverageMatrix   The type of the matrix that is used to store how often individual examples and labels
     *                          have been covered
     */
    template<typename LabelMatrix, typename CoverageMatrix>
    class MLRLSECO_API DenseDecomposableStatisticMatrix {
        public:

            /**
             * A view that provides access to the data structures, a `DenseDecomposableStatisticMatrix` is backed by.
             */
            struct View {
                public:

                    /**
                     * @param labelMatrix           A reference to an object of template type `LabelMatrix` that
                     *                              provides access to the labels of the training examples
                     * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that
                     *                              stores the predictions of the default rule
                     * @param coverageMatrix        A reference to an object of template type `CoverageMatrix` that
                     *                              stores how often individual examples and labels have been covered
                     */
                    View(const LabelMatrix& labelMatrix, const BinarySparseArrayVector& majorityLabelVector,
                         CoverageMatrix& coverageMatrix)
                        : labelMatrix(labelMatrix), majorityLabelVector(majorityLabelVector),
                          coverageMatrix(coverageMatrix) {}

                    /**
                     * A reference to an object of template type `LabelMatrix` that provides access to the labels of the
                     * training examples.
                     */
                    const LabelMatrix& labelMatrix;

                    /**
                     * A reference to an object of type `BinarySparseArrayVector` that stores the predictions of the
                     * default rule.
                     */
                    const BinarySparseArrayVector& majorityLabelVector;

                    /**
                     * A reference to an object of template type `CoverageMatrix` that stores how often individual
                     * examples and labels have been covered.
                     */
                    CoverageMatrix& coverageMatrix;
            };

            /**
             * A reference to an object of template type `LabelMatrix` that provides access to the labels of the
             * training examples.
             */
            const LabelMatrix& labelMatrix;

            /**
             * An unique pointer to an object of type `BinarySparseArrayVector` that stores the predictions of the
             * default rule.
             */
            const std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr;

            /**
             * An unique pointer to an object of template type `CoverageMatrix` that stores how often individual
             * examples and labels have been covered.
             */
            std::unique_ptr<CoverageMatrix> coverageMatrixPtr;

        public:

            /**
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param majorityLabelVectorPtr    An unique pointer to an object of type `BinarySparseArrayVector` that
             *                                  stores the predictions of the default rule
             * @param coverageMatrixPtr         An unique pointer to an object of template type `CoverageMatrix` that
             *                                  stores how often individual examples and labels have been covered
             */
            DenseDecomposableStatisticMatrix(const LabelMatrix& labelMatrix,
                                             std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr,
                                             std::unique_ptr<CoverageMatrix> coverageMatrixPtr);

            /**
             * Returns the number of rows in the matrix.
             *
             * @return The number of rows in the matrix
             */
            uint32 getNumRows() const;

            /**
             * Returns the number of columns in the matrix.
             *
             * @return The number of columns in the matrix
             */
            uint32 getNumCols() const;

            /**
             * Returns a reference to the view, that provides access to the data structures this matrix is backed by.
             *
             * @return A reference to an object of type `View`
             */
            View getView();
    };

}
