/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/model/rule_model.hpp"

/**
 * An implementation of the type `IRuleModel` that stores several rules in an in-memory HDF5 file. Optionally, the model
 * may also contain a default rule that either takes precedence over the remaining rules or not.
 */
class Hdf5RuleModel : public IRuleModel {
    public:

        /**
         * @param defaultRuleTakesPrecedence True, if the default rule should take precedence over the remaining rules,
         *                                   false otherwise
         */
        explicit Hdf5RuleModel(bool defaultRuleTakesPrecedence);

        uint32 getNumRules() const override;

        uint32 getNumUsedRules() const override;

        void setNumUsedRules(uint32 numUsedRules) override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousView<const float32>& featureMatrix,
                                                              const IOutputSpaceInfo& outputSpaceInfo,
                                                              uint32 numOutputs) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrView<const float32>& featureMatrix,
                                                              const IOutputSpaceInfo& outputSpaceInfo,
                                                              uint32 numOutputs) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
          const IOutputSpaceInfo& outputSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;
};
