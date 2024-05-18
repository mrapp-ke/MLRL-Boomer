/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_lil_binary.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"

#include <memory>

/**
 * A two-dimensional view that provides row-wise access to binary values stored in a matrix in the compressed sparse row
 * (CSR) format that have been copied from a `BinaryLilMatrix`.
 */
class MLRLCOMMON_API BinarySparsePredictionView final : public AllocatedBinaryCsrView {
    public:

        /**
         * @param lilMatrix         A reference to an object of type `BinaryLilMatrix` to be copied
         * @param numCols           The number of columns of the given `BinaryLilMatrix`
         * @param numDenseElements  The number of dense elements explicitly stored in the given `BinaryLilMatrix`
         */
        BinarySparsePredictionView(const BinaryLilMatrix& lilMatrix, uint32 numCols, uint32 numDenseElements);

        /**
         * @param other A reference to an object of type `BinarySparsePredictionView` that should be moved
         */
        BinarySparsePredictionView(BinarySparsePredictionView&& other);
};

/**
 * A sparse matrix that provides read-only access to binary predictions that are stored in the compressed sparse row
 * (CSR) format.
 */
class MLRLCOMMON_API BinarySparsePredictionMatrix final
    : public IterableBinarySparseMatrixDecorator<MatrixDecorator<BinarySparsePredictionView>> {
    public:

        /**
         * @param lilMatrix         A reference to an object of type `BinaryLilMatrix` to be copied
         * @param numCols           The number of columns in the given `BinaryLilMatrix`
         * @param numDenseElements  The number of dense elements explicitly stored in the given `BinaryLilMatrix`
         */
        BinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix, uint32 numCols, uint32 numDenseElements);

        /**
         * Returns a pointer to the array that stores the column indices of all dense elements explicitly stored in the
         * matrix.
         *
         * @return A pointer to the array that stores the column indices of all dense elements explicitly stored in the
         *         matrix
         */
        uint32* getIndices();

        /**
         * Releases the ownership of the array that stores the column indices of all dense elements explicitly stored in
         * the matrix. As a result, the behavior of this matrix becomes undefined and it should not be used anymore. The
         * caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return A pointer to the array that stores the column indices of all dense elements explicitly stored in the
         *         matrix
         */
        uint32* releaseIndices();

        /**
         * Returns a pointer to the array that stores the indices of the first dense element that corresponds to a
         * certain row.
         *
         * @return A pointer to the array that stores the indices of the first dense element that corresponds to a
         *         certain row
         */
        uint32* getIndptr();

        /**
         * Releases the ownership of the array that stores the indices of the first dense element that corresponds to a
         * certain row. As a result, the behavior of this matrix becomes undefined and it should not be used anymore.
         * The caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return A pointer to an array that stores the indices of the first dense element that corresponds to a
         *         certain row
         */
        uint32* releaseIndptr();
};

/**
 * Creates and returns a new object of the type `BinarySparsePredictionMatrix` as a copy of an existing
 * `BinaryLilMatrix`.
 *
 * @param lilMatrix         A reference to an object of type `BinaryLilMatrix` to be copied
 * @param numCols           The number of columns in the given `BinaryLilMatrix`
 * @param numDenseElements  The number of dense elements explicitly stored in the given `BinaryLilMatrix`
 * @return                  An unique pointer to an object of type `BinarySparsePredictionMatrix` that has been created
 */
std::unique_ptr<BinarySparsePredictionMatrix> createBinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix,
                                                                                 uint32 numCols,
                                                                                 uint32 numDenseElements);
