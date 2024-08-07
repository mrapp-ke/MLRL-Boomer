#include "mlrl/common/input/feature_matrix_csc.hpp"

#include "mlrl/common/data/view_matrix_csc.hpp"

/**
 * An implementation of the type `ICscFeatureMatrix` that provides column-wise read-only access to the feature values of
 * examples that are stored in a pre-allocated sparse matrix in the compressed sparse column (CSC) format.
 */
class CscFeatureMatrix final : public IterableSparseMatrixDecorator<MatrixDecorator<CscView<const float32>>>,
                               public ICscFeatureMatrix {
    public:

        /**
         * @param values        A pointer to an array of type `float32`, shape `(numDenseElements)`, that stores the
         *                      values of all dense elements explicitly stored in the matrix
         * @param indices       A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the
         *                      row-indices, the values in `values` correspond to
         * @param indptr        A pointer to an array of type `uint32`, shape `(numCols + 1)`, that stores the indices
         *                      of the first element in `values` and `indices` that corresponds to a certain column. The
         *                      index at the last position is equal to `numDenseElements`
         * @param numRows       The number of rows in the feature matrix
         * @param numCols       The number of columns in the feature matrix
         * @param sparseValue   The value that should be used for sparse elements in the feature matrix
         */
        CscFeatureMatrix(const float32* values, uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols,
                         float32 sparseValue)
            : IterableSparseMatrixDecorator<MatrixDecorator<CscView<const float32>>>(
                CscView<const float32>(values, indices, indptr, numRows, numCols, sparseValue)) {}

        bool isSparse() const override {
            return true;
        }

        uint32 getNumExamples() const override {
            return this->getNumRows();
        }

        uint32 getNumFeatures() const override {
            return this->getNumCols();
        }

        std::unique_ptr<IFeatureVector> createFeatureVector(uint32 featureIndex,
                                                            const IFeatureType& featureType) const override {
            return featureType.createFeatureVector(featureIndex, this->getView());
        }
};

std::unique_ptr<ICscFeatureMatrix> createCscFeatureMatrix(const float32* values, uint32* indices, uint32* indptr,
                                                          uint32 numRows, uint32 numCols, float32 sparseValue) {
    return std::make_unique<CscFeatureMatrix>(values, indices, indptr, numRows, numCols, sparseValue);
}
