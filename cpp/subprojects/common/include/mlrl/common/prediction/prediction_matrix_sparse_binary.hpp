/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/matrix_lil_binary.hpp"
#include "mlrl/common/data/view_matrix_csr_binary.hpp"
#include "mlrl/common/util/dll_exports.hpp"

#include <memory>

/**
 * A sparse matrix that provides read-only access to binary predictions that are stored in the compressed sparse row
 * (CSR) format.
 *
 * The matrix maintains two arrays, referred to as `indptr` and `colIndices`. The latter stores a column-index for each
 * of the `numNonZeroValues` non-zero elements in the matrix. The former stores `numRows + 1` row-indices that specify
 * the first element in `colIndices` that correspond to a certain row. The index at the last position is equal to the
 * number of non-zero values in the matrix.
 */
class MLRLCOMMON_API BinarySparsePredictionMatrix final : public BinaryCsrView {
    private:

        uint32* colIndices_;

        uint32* indptr_;

    public:

        /**
         * @param indices   A pointer to an array of type `uint32`, shape `(numNonZeroValues)`, that stores the
         *                  column-indices, the non-zero elements correspond to
         * @param indptr    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of
         *                  the first element in `indices` that corresponds to a certain row. The index at the last
         *                  position is equal to `numNonZeroValues`
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        BinarySparsePredictionMatrix(uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols);

        ~BinarySparsePredictionMatrix() override;

        /**
         * Returns a pointer to the array `colIndices`.
         *
         * @return A pointer to the array `colIndices`
         */
        uint32* getColIndices();

        /**
         * Releases the ownership of the array `colIndices`. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array `colIndices`
         */
        uint32* releaseColIndices();

        /**
         * Returns a pointer to the array `indptr`.
         *
         * @return A pointer to the array `indptr`
         */
        uint32* getIndptr();

        /**
         * Releases the ownership of the array `indptr`. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array `indptr`
         */
        uint32* releaseIndptr();
};

/**
 * Creates and returns a new object of the type `BinarySparsePredictionMatrix` as a copy of an existing
 * `BinaryLilMatrix`.
 *
 * @param lilMatrix             A reference to an object of type `BinaryLilMatrix` to be copied
 * @param numCols               The number of columns of the given `BinaryLilMatrix`
 * @param numNonZeroElements    The number of non-zero elements in the given `BinaryLilMatrix`
 * @return                      An unique pointer to an object of type `BinarySparsePredictionMatrix` that has been
 *                              created
 */
std::unique_ptr<BinarySparsePredictionMatrix> createBinarySparsePredictionMatrix(const BinaryLilMatrix& lilMatrix,
                                                                                 uint32 numCols,
                                                                                 uint32 numNonZeroElements);
