#include "mlrl/common/input/feature_matrix_c_contiguous.hpp"

#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"

/**
 * An implementation of the type `ICContiguousFeatureMatrix` that provides row-wise read-only access to the feature
 * values of examples that are stored in a C-contiguous array.
 */
class CContiguousFeatureMatrix final : public MatrixDecorator<CContiguousView<const float32>>,
                                       public ICContiguousFeatureMatrix {
    public:

        /**
         * @param array     A pointer to a C-contiguous array of type `float32` that stores the values, the feature
         *                  matrix provides access to
         * @param numRows   The number of rows in the feature matrix
         * @param numCols   The number of columns in the feature matrix
         */
        CContiguousFeatureMatrix(const float32* array, uint32 numRows, uint32 numCols)
            : MatrixDecorator<CContiguousView<const float32>>(CContiguousView<const float32>(array, numRows, numCols)) {
        }

        bool isSparse() const override {
            return false;
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

std::unique_ptr<ICContiguousFeatureMatrix> createCContiguousFeatureMatrix(const float32* array, uint32 numRows,
                                                                          uint32 numCols) {
    return std::make_unique<CContiguousFeatureMatrix>(array, numRows, numCols);
}
