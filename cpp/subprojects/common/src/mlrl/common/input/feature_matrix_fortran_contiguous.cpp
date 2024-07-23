#include "mlrl/common/input/feature_matrix_fortran_contiguous.hpp"

#include "mlrl/common/data/matrix_dense.hpp"
#include "mlrl/common/data/view_matrix_fortran_contiguous.hpp"

/**
 * An implementation of the type `IFortranContiguousFeatureMatrix` that provides column-wise read-only access to the
 * feature values of examples that are stored in a pre-allocated Fortran-contiguous array.
 */
class FortranContiguousFeatureMatrix final : public DenseMatrixDecorator<FortranContiguousView<const float32>>,
                                             public IFortranContiguousFeatureMatrix {
    public:

        /**
         * @param numRows   The number of rows in the feature matrix
         * @param numCols   The number of columns in the feature matrix
         * @param array     A pointer to a Fortran-contiguous array of type `float32` that stores the feature values
         */
        FortranContiguousFeatureMatrix(const float32* array, uint32 numRows, uint32 numCols)
            : DenseMatrixDecorator<FortranContiguousView<const float32>>(
                FortranContiguousView<const float32>(array, numRows, numCols)) {}

        bool isSparse() const override {
            return false;
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

std::unique_ptr<IFortranContiguousFeatureMatrix> createFortranContiguousFeatureMatrix(const float32* array,
                                                                                      uint32 numRows, uint32 numCols) {
    return std::make_unique<FortranContiguousFeatureMatrix>(array, numRows, numCols);
}
