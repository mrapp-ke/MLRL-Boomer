#include "common/learner.hpp"


/**
 * An implementation of the type `ITrainingResult` that provides access to the result of training an
 * `AbstractRuleLearner`.
 */
class TrainingResult final : public ITrainingResult {

    private:

        uint32 numLabels_;

        std::unique_ptr<IRuleModel> ruleModelPtr_;

        std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr_;

    public:

        /**
         * @param numLabels         The number of labels for which a model has been trained
         * @param ruleModelPtr      An unique pointer to an object of type `IRuleModel` that has been trained
         * @param labelSpaceInfoPtr An unique pointer to an object of type `ILabelSpaceInfo` that may be used as a basis
         *                          for making predictions
         */
        TrainingResult(uint32 numLabels, std::unique_ptr<IRuleModel> ruleModelPtr,
                       std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr)
            : numLabels_(numLabels), ruleModelPtr_(std::move(ruleModelPtr)),
              labelSpaceInfoPtr_(std::move(labelSpaceInfoPtr)) {

        }

        uint32 getNumLabels() const override {
            return numLabels_;
        }

        std::unique_ptr<IRuleModel>& getRuleModel() override {
            return ruleModelPtr_;
        }

        const std::unique_ptr<IRuleModel>& getRuleModel() const override {
            return ruleModelPtr_;
        }

        std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() override {
            return labelSpaceInfoPtr_;
        }

        const std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() const override {
            return labelSpaceInfoPtr_;
        }

};

std::unique_ptr<IRuleModelAssemblageFactory> AbstractRuleLearner::createRuleModelAssemblageFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IThresholdsFactory> AbstractRuleLearner::createThresholdsFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IRuleInductionFactory> AbstractRuleLearner::createRuleInductionFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<ILabelSamplingFactory> AbstractRuleLearner::createLabelSamplingFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IInstanceSamplingFactory> AbstractRuleLearner::createInstanceSamplingFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IFeatureSamplingFactory> AbstractRuleLearner::createFeatureSamplingFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IPartitionSamplingFactory> AbstractRuleLearner::createPartitionSamplingFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IPruningFactory> AbstractRuleLearner::createPruningFactory() const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<IPostProcessorFactory> AbstractRuleLearner::createPostProcessorFactory() const {
    // TODO Implement
    return nullptr;
}

bool AbstractRuleLearner::useDefaultRule() const {
    return true;
}

void AbstractRuleLearner::createStoppingCriterionFactories(
        std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const {
    // TODO Implement
}

std::unique_ptr<IRegressionPredictorFactory> AbstractRuleLearner::createRegressionPredictorFactory() const {
    return nullptr;
}

std::unique_ptr<IProbabilityPredictorFactory> AbstractRuleLearner::createProbabilityPredictorFactory() const {
    return nullptr;
}

std::unique_ptr<ITrainingResult> AbstractRuleLearner::fit(
        const INominalFeatureMask& nominalFeatureMask, const IColumnWiseFeatureMatrix& featureMatrix,
        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<DensePredictionMatrix<uint8>> AbstractRuleLearner::predictLabels(
        const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictLabels(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                               trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<uint8>> AbstractRuleLearner::predictLabels(
        const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
        uint32 numLabels) const {
    // TODO Implement
    return nullptr;
}

std::unique_ptr<BinarySparsePredictionMatrix> AbstractRuleLearner::predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictSparseLabels(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                                     trainingResult.getNumLabels());
}

std::unique_ptr<BinarySparsePredictionMatrix> AbstractRuleLearner::predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    // TODO Implement
    return nullptr;
}

bool AbstractRuleLearner::canPredictScores() const {
    // TODO Implement
    return false;
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictScores(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                               trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    // TODO Implement
    return nullptr;
}

bool AbstractRuleLearner::canPredictProbabilities() const {
    // TODO Implement
    return false;
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictProbabilities(featureMatrix, *trainingResult.getRuleModel(),
                                      *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    // TODO Implement
    return nullptr;
}
