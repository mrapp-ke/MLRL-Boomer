#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

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

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override {
            value_const_iterator columnIterator = this->values_cbegin(featureIndex);
            uint32 numElements = this->getNumRows();
            featureVectorPtr = std::make_unique<FeatureVector>(numElements);
            FeatureVector::iterator vectorIterator = featureVectorPtr->begin();
            uint32 i = 0;

            for (uint32 j = 0; j < numElements; j++) {
                float32 value = columnIterator[j];

                if (std::isnan(value)) {
                    featureVectorPtr->addMissingIndex(j);
                } else {
                    vectorIterator[i].index = j;
                    vectorIterator[i].value = value;
                    i++;
                }
            }

            featureVectorPtr->setNumElements(i, true);
        }
};

std::unique_ptr<IFortranContiguousFeatureMatrix> createFortranContiguousFeatureMatrix(const float32* array,
                                                                                      uint32 numRows, uint32 numCols) {
    return std::make_unique<FortranContiguousFeatureMatrix>(array, numRows, numCols);
}

#ifdef _WIN32
    #pragma warning(pop)
#endif
