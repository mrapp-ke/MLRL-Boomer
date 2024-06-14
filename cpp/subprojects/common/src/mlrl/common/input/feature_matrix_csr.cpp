#include "mlrl/common/input/feature_matrix_csr.hpp"

#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"

/**
 * An implementation of the type `ICsrFeatureMatrix` that provides row-wise read-only access to the feature values of
 * examples that are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrFeatureMatrix final : public MatrixDecorator<CsrView<const float32>>,
                               public ICsrFeatureMatrix {
    public:

        /**
         * @param values      A pointer to an array of type `float32`, shape `(numDenseElements)`, that stores the
         *                    values of all dense elements explicitly stored in the matrix
         * @param indices     A pointer to an array of type `uint32`, shape `(numDenseElements)`, that stores the
         *                    column-indices, the values in `values` correspond to
         * @param indptr      A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices of
         *                    the first element in `values` and `indices` that corresponds to a certain row. The index
         *                    at the last position is equal to `numDenseElements`
         * @param numRows     The number of rows in the feature matrix
         * @param numCols     The number of columns in the feature matrix
         * @param sparseValue The value that should be used for sparse elements in the feature matrix
         */
        CsrFeatureMatrix(const float32* values, uint32* indices, uint32* indptr, uint32 numRows, uint32 numCols,
                         float32 sparseValue)
            : MatrixDecorator<CsrView<const float32>>(
                CsrView<const float32>(values, indices, indptr, numRows, numCols, sparseValue)) {}

        bool isSparse() const override {
            return true;
        }

        uint32 getNumExamples() const override {
            return this->getNumRows();
        }

        uint32 getNumFeatures() const override {
            return this->getNumCols();
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return ruleModel.createBinaryPredictor(factory, this->getView(), outputSpaceInfo,
                                                   marginalProbabilityCalibrationModel,
                                                   jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return ruleModel.createSparseBinaryPredictor(factory, this->getView(), outputSpaceInfo,
                                                         marginalProbabilityCalibrationModel,
                                                         jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const IRuleModel& ruleModel,
                                                              const IOutputSpaceInfo& outputSpaceInfo,
                                                              uint32 numOutputs) const override {
            return ruleModel.createScorePredictor(factory, this->getView(), outputSpaceInfo, numOutputs);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return ruleModel.createProbabilityPredictor(factory, this->getView(), outputSpaceInfo,
                                                        marginalProbabilityCalibrationModel,
                                                        jointProbabilityCalibrationModel, numLabels);
        }
};

std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(const float32* values, uint32* indices, uint32* indptr,
                                                          uint32 numRows, uint32 numCols, float32 sparseValue) {
    return std::make_unique<CsrFeatureMatrix>(values, indices, indptr, numRows, numCols, sparseValue);
}
