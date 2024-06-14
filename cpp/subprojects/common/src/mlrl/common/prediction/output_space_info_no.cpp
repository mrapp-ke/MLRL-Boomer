#include "mlrl/common/prediction/output_space_info_no.hpp"

#include "mlrl/common/model/rule_list.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"

/**
 * An implementation of the type `INoOutputSpaceInfo` that does not provide any information about the output space.
 */
class NoOutputSpaceInfo final : public INoOutputSpaceInfo {
    public:

        std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator(
          const IJointProbabilityCalibratorFactory& factory,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override {
            return factory.create(marginalProbabilityCalibrationModel, nullptr);
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousView<const float32>& featureMatrix,
                                                              const RuleList& model, uint32 numOutputs) const override {
            return factory.create(featureMatrix, model, nullptr, numOutputs);
        }

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrView<const float32>& featureMatrix,
                                                              const RuleList& model, uint32 numOutputs) const override {
            return factory.create(featureMatrix, model, nullptr, numOutputs);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override {
            return factory.create(featureMatrix, model, nullptr, marginalProbabilityCalibrationModel,
                                  jointProbabilityCalibrationModel, numLabels);
        }
};

std::unique_ptr<INoOutputSpaceInfo> createNoOutputSpaceInfo() {
    return std::make_unique<NoOutputSpaceInfo>();
}
