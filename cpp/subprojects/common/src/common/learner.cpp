#include "common/learner.hpp"
#include "common/binning/feature_binning_no.hpp"
#include "common/multi_threading/multi_threading_no.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"
#include "common/rule_pruning/rule_pruning_no.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/label_sampling_no.hpp"
#include "common/sampling/partition_sampling_no.hpp"
#include "common/stopping/stopping_criterion_size.hpp"
#include "common/util/validation.hpp"


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

AbstractRuleLearner::Config::Config(RuleCompareFunction ruleCompareFunction)
    : unusedRuleRemovalConfigPtr_(std::make_unique<UnusedRuleRemovalConfig>()),
      ruleCompareFunction_(ruleCompareFunction) {
    this->useDefaultRule();
    this->useSequentialRuleModelAssemblage();
    this->useGreedyTopDownRuleInduction();
    this->useNoFeatureBinning();
    this->useNoLabelSampling();
    this->useNoInstanceSampling();
    this->useNoFeatureSampling();
    this->useNoPartitionSampling();
    this->useNoRulePruning();
    this->useNoPostProcessor();
    this->useNoParallelRuleRefinement();
    this->useNoParallelStatisticUpdate();
    this->useNoParallelPrediction();
    this->useNoSizeStoppingCriterion();
    this->useNoTimeStoppingCriterion();
    this->useNoGlobalPruning();
    this->useNoSequentialPostOptimization();
}

RuleCompareFunction AbstractRuleLearner::Config::getRuleCompareFunction() const {
    return ruleCompareFunction_;
}

std::unique_ptr<IDefaultRuleConfig>& AbstractRuleLearner::Config::getDefaultRuleConfigPtr() {
    return defaultRuleConfigPtr_;
}

std::unique_ptr<IRuleModelAssemblageConfig>& AbstractRuleLearner::Config::getRuleModelAssemblageConfigPtr() {
    return ruleModelAssemblageConfigPtr_;
}

std::unique_ptr<IRuleInductionConfig>& AbstractRuleLearner::Config::getRuleInductionConfigPtr() {
    return ruleInductionConfigPtr_;
}

std::unique_ptr<IFeatureBinningConfig>& AbstractRuleLearner::Config::getFeatureBinningConfigPtr() {
    return featureBinningConfigPtr_;
}

std::unique_ptr<ILabelSamplingConfig>& AbstractRuleLearner::Config::getLabelSamplingConfigPtr() {
    return labelSamplingConfigPtr_;
}

std::unique_ptr<IInstanceSamplingConfig>& AbstractRuleLearner::Config::getInstanceSamplingConfigPtr() {
    return instanceSamplingConfigPtr_;
}

std::unique_ptr<IFeatureSamplingConfig>& AbstractRuleLearner::Config::getFeatureSamplingConfigPtr() {
    return featureSamplingConfigPtr_;
}

std::unique_ptr<IPartitionSamplingConfig>& AbstractRuleLearner::Config::getPartitionSamplingConfigPtr() {
    return partitionSamplingConfigPtr_;
}

std::unique_ptr<IRulePruningConfig>& AbstractRuleLearner::Config::getRulePruningConfigPtr() {
    return rulePruningConfigPtr_;
}

std::unique_ptr<IPostProcessorConfig>& AbstractRuleLearner::Config::getPostProcessorConfigPtr() {
    return postProcessorConfigPtr_;
}

std::unique_ptr<IMultiThreadingConfig>& AbstractRuleLearner::Config::getParallelRuleRefinementConfigPtr() {
    return parallelRuleRefinementConfigPtr_;
}

std::unique_ptr<IMultiThreadingConfig>& AbstractRuleLearner::Config::getParallelStatisticUpdateConfigPtr() {
    return parallelStatisticUpdateConfigPtr_;
}

std::unique_ptr<IMultiThreadingConfig>& AbstractRuleLearner::Config::getParallelPredictionConfigPtr() {
    return parallelPredictionConfigPtr_;
}

std::unique_ptr<SizeStoppingCriterionConfig>& AbstractRuleLearner::Config::getSizeStoppingCriterionConfigPtr() {
    return sizeStoppingCriterionConfigPtr_;
}

std::unique_ptr<TimeStoppingCriterionConfig>& AbstractRuleLearner::Config::getTimeStoppingCriterionConfigPtr() {
    return timeStoppingCriterionConfigPtr_;
}

std::unique_ptr<IGlobalPruningConfig>& AbstractRuleLearner::Config::getGlobalPruningConfigPtr() {
    return globalPruningConfigPtr_;
}

std::unique_ptr<SequentialPostOptimizationConfig>& AbstractRuleLearner::Config::getSequentialPostOptimizationConfigPtr() {
    return sequentialPostOptimizationConfigPtr_;
}

std::unique_ptr<UnusedRuleRemovalConfig>& AbstractRuleLearner::Config::getUnusedRuleRemovalConfigPtr() {
    return unusedRuleRemovalConfigPtr_;
}

void AbstractRuleLearner::Config::useDefaultRule() {
    defaultRuleConfigPtr_ = std::make_unique<DefaultRuleConfig>(true);
}

void AbstractRuleLearner::Config::useSequentialRuleModelAssemblage() {
    ruleModelAssemblageConfigPtr_ = std::make_unique<SequentialRuleModelAssemblageConfig>(defaultRuleConfigPtr_);
}

IGreedyTopDownRuleInductionConfig& AbstractRuleLearner::Config::useGreedyTopDownRuleInduction() {
    std::unique_ptr<GreedyTopDownRuleInductionConfig> ptr =
        std::make_unique<GreedyTopDownRuleInductionConfig>(ruleCompareFunction_, parallelRuleRefinementConfigPtr_);
    IGreedyTopDownRuleInductionConfig& ref = *ptr;
    ruleInductionConfigPtr_ = std::move(ptr);
    return ref;
}

void AbstractRuleLearner::Config::useNoFeatureBinning() {
    featureBinningConfigPtr_ = std::make_unique<NoFeatureBinningConfig>(parallelStatisticUpdateConfigPtr_);
}

void AbstractRuleLearner::Config::useNoLabelSampling() {
    labelSamplingConfigPtr_ = std::make_unique<NoLabelSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoInstanceSampling() {
    instanceSamplingConfigPtr_ = std::make_unique<NoInstanceSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoFeatureSampling() {
    featureSamplingConfigPtr_ = std::make_unique<NoFeatureSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoPartitionSampling() {
    partitionSamplingConfigPtr_ = std::make_unique<NoPartitionSamplingConfig>();
}

void AbstractRuleLearner::Config::useNoRulePruning() {
    rulePruningConfigPtr_ = std::make_unique<NoRulePruningConfig>();
}

void AbstractRuleLearner::Config::useNoPostProcessor() {
    postProcessorConfigPtr_ = std::make_unique<NoPostProcessorConfig>();
}

void AbstractRuleLearner::Config::useNoParallelRuleRefinement() {
    parallelRuleRefinementConfigPtr_ = std::make_unique<NoMultiThreadingConfig>();
}

void AbstractRuleLearner::Config::useNoParallelStatisticUpdate() {
    parallelStatisticUpdateConfigPtr_ = std::make_unique<NoMultiThreadingConfig>();
}

void AbstractRuleLearner::Config::useNoParallelPrediction() {
    parallelPredictionConfigPtr_ = std::make_unique<NoMultiThreadingConfig>();
}

void AbstractRuleLearner::Config::useNoSizeStoppingCriterion() {
    sizeStoppingCriterionConfigPtr_ = nullptr;
}

void AbstractRuleLearner::Config::useNoTimeStoppingCriterion() {
    timeStoppingCriterionConfigPtr_ = nullptr;
}

void AbstractRuleLearner::Config::useNoGlobalPruning() {
    globalPruningConfigPtr_ = nullptr;
}

void AbstractRuleLearner::Config::useNoSequentialPostOptimization() {
    sequentialPostOptimizationConfigPtr_ = nullptr;
}

AbstractRuleLearner::AbstractRuleLearner(IRuleLearner::IConfig& config)
    : config_(config) {

}

std::unique_ptr<IRuleModelAssemblageFactory> AbstractRuleLearner::createRuleModelAssemblageFactory(
        const IRowWiseLabelMatrix& labelMatrix) const {
    return config_.getRuleModelAssemblageConfigPtr()->createRuleModelAssemblageFactory(labelMatrix);
}

std::unique_ptr<IThresholdsFactory> AbstractRuleLearner::createThresholdsFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    return config_.getFeatureBinningConfigPtr()->createThresholdsFactory(featureMatrix, labelMatrix);
}

std::unique_ptr<IRuleInductionFactory> AbstractRuleLearner::createRuleInductionFactory(
        const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    return config_.getRuleInductionConfigPtr()->createRuleInductionFactory(featureMatrix, labelMatrix);
}

std::unique_ptr<ILabelSamplingFactory> AbstractRuleLearner::createLabelSamplingFactory(
        const ILabelMatrix& labelMatrix) const {
    return config_.getLabelSamplingConfigPtr()->createLabelSamplingFactory(labelMatrix);
}

std::unique_ptr<IInstanceSamplingFactory> AbstractRuleLearner::createInstanceSamplingFactory() const {
    return config_.getInstanceSamplingConfigPtr()->createInstanceSamplingFactory();
}

std::unique_ptr<IFeatureSamplingFactory> AbstractRuleLearner::createFeatureSamplingFactory(
        const IFeatureMatrix& featureMatrix) const {
    return config_.getFeatureSamplingConfigPtr()->createFeatureSamplingFactory(featureMatrix);
}

std::unique_ptr<IPartitionSamplingFactory> AbstractRuleLearner::createPartitionSamplingFactory() const {
    return config_.getPartitionSamplingConfigPtr()->createPartitionSamplingFactory();
}

std::unique_ptr<IRulePruningFactory> AbstractRuleLearner::createRulePruningFactory() const {
    return config_.getRulePruningConfigPtr()->createRulePruningFactory();
}

std::unique_ptr<IPostProcessorFactory> AbstractRuleLearner::createPostProcessorFactory() const {
    return config_.getPostProcessorConfigPtr()->createPostProcessorFactory();
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createSizeStoppingCriterionFactory() const {
    std::unique_ptr<SizeStoppingCriterionConfig>& configPtr = config_.getSizeStoppingCriterionConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createTimeStoppingCriterionFactory() const {
    std::unique_ptr<TimeStoppingCriterionConfig>& configPtr = config_.getTimeStoppingCriterionConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IStoppingCriterionFactory> AbstractRuleLearner::createGlobalPruningFactory() const {
    std::unique_ptr<IGlobalPruningConfig>& configPtr = config_.getGlobalPruningConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createStoppingCriterionFactory() : nullptr;
}

std::unique_ptr<IPostOptimizationPhaseFactory> AbstractRuleLearner::createSequentialPostOptimizationFactory() const {
    std::unique_ptr<SequentialPostOptimizationConfig>& configPtr = config_.getSequentialPostOptimizationConfigPtr();
    return configPtr.get() != nullptr ? configPtr->createPostOptimizationPhaseFactory() : nullptr;
}

std::unique_ptr<IPostOptimizationPhaseFactory> AbstractRuleLearner::createUnusedRuleRemovalFactory() const {
    std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr = config_.getGlobalPruningConfigPtr();

    if (globalPruningConfigPtr && globalPruningConfigPtr->shouldRemoveUnusedRules()) {
        std::unique_ptr<UnusedRuleRemovalConfig>& configPtr = config_.getUnusedRuleRemovalConfigPtr();
        return configPtr->createPostOptimizationPhaseFactory();
    }

    return nullptr;
}

void AbstractRuleLearner::createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const {
    std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactory = this->createSizeStoppingCriterionFactory();

    if (stoppingCriterionFactory) {
        factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
    }

    stoppingCriterionFactory = this->createTimeStoppingCriterionFactory();

    if (stoppingCriterionFactory) {
        factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
    }

    stoppingCriterionFactory = this->createGlobalPruningFactory();

    if (stoppingCriterionFactory) {
        factory.addStoppingCriterionFactory(std::move(stoppingCriterionFactory));
    }
}

void AbstractRuleLearner::createPostOptimizationPhaseFactories(PostOptimizationPhaseListFactory& factory) const {
    std::unique_ptr<IPostOptimizationPhaseFactory> postOptimizationPhaseFactory =
        this->createUnusedRuleRemovalFactory();

    if (postOptimizationPhaseFactory) {
        factory.addPostOptimizationPhaseFactory(std::move(postOptimizationPhaseFactory));
    }

    postOptimizationPhaseFactory = this->createSequentialPostOptimizationFactory();

    if (postOptimizationPhaseFactory) {
        factory.addPostOptimizationPhaseFactory(std::move(postOptimizationPhaseFactory));
    }
}

std::unique_ptr<IClassificationPredictorFactory> AbstractRuleLearner::createClassificationPredictorFactory(
        const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return nullptr;
}

std::unique_ptr<IRegressionPredictorFactory> AbstractRuleLearner::createRegressionPredictorFactory(
        const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return nullptr;
}

std::unique_ptr<IProbabilityPredictorFactory> AbstractRuleLearner::createProbabilityPredictorFactory(
        const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return nullptr;
}

std::unique_ptr<ITrainingResult> AbstractRuleLearner::fit(const IFeatureInfo& featureInfo,
                                                          const IColumnWiseFeatureMatrix& featureMatrix,
                                                          const IRowWiseLabelMatrix& labelMatrix,
                                                          uint32 randomState) const {
    assertGreaterOrEqual<uint32>("randomState", randomState, 1);

    // Create stopping criteria...
    std::unique_ptr<StoppingCriterionListFactory> stoppingCriterionFactoryPtr =
        std::make_unique<StoppingCriterionListFactory>();
    this->createStoppingCriterionFactories(*stoppingCriterionFactoryPtr);

    // Create post-optimization phases...
    std::unique_ptr<PostOptimizationPhaseListFactory> postOptimizationFactoryPtr =
        std::make_unique<PostOptimizationPhaseListFactory>();
    this->createPostOptimizationPhaseFactories(*postOptimizationFactoryPtr);

    std::unique_ptr<ILabelSpaceInfo> labelSpaceInfoPtr = this->createLabelSpaceInfo(labelMatrix);
    std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr =
        this->createRuleModelAssemblageFactory(labelMatrix);
    std::unique_ptr<IRuleModelAssemblage> ruleModelAssemblagePtr = ruleModelAssemblageFactoryPtr->create(
        this->createModelBuilderFactory(), this->createStatisticsProviderFactory(featureMatrix, labelMatrix),
        this->createThresholdsFactory(featureMatrix, labelMatrix),
        this->createRuleInductionFactory(featureMatrix, labelMatrix), this->createLabelSamplingFactory(labelMatrix),
        this->createInstanceSamplingFactory(), this->createFeatureSamplingFactory(featureMatrix),
        this->createPartitionSamplingFactory(), this->createRulePruningFactory(), this->createPostProcessorFactory(),
        std::move(postOptimizationFactoryPtr), std::move(stoppingCriterionFactoryPtr));
    std::unique_ptr<IRuleModel> ruleModelPtr = ruleModelAssemblagePtr->induceRules(featureInfo, featureMatrix,
                                                                                   labelMatrix, randomState);
    return std::make_unique<TrainingResult>(labelMatrix.getNumCols(), std::move(ruleModelPtr),
                                            std::move(labelSpaceInfoPtr));
}

bool AbstractRuleLearner::canPredictLabels(const IRowWiseFeatureMatrix& featureMatrix,
                                           const ITrainingResult& trainingResult) const {
    return this->canPredictLabels(featureMatrix, trainingResult.getNumLabels());
}

bool AbstractRuleLearner::canPredictLabels(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return this->createClassificationPredictorFactory(featureMatrix, numLabels) != nullptr;
}

std::unique_ptr<DensePredictionMatrix<uint8>> AbstractRuleLearner::predictLabels(
        const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictLabels(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                               trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<uint8>> AbstractRuleLearner::predictLabels(
        const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
        uint32 numLabels) const {
    std::unique_ptr<IClassificationPredictorFactory> predictorFactoryPtr =
        this->createClassificationPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        std::unique_ptr<IClassificationPredictor> predictorPtr =
            ruleModel.createClassificationPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictLabels(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict binary labels");
}

std::unique_ptr<BinarySparsePredictionMatrix> AbstractRuleLearner::predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictSparseLabels(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                                     trainingResult.getNumLabels());
}

std::unique_ptr<BinarySparsePredictionMatrix> AbstractRuleLearner::predictSparseLabels(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IClassificationPredictorFactory> predictorFactoryPtr =
        this->createClassificationPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        std::unique_ptr<IClassificationPredictor> predictorPtr =
            ruleModel.createClassificationPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictSparseLabels(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict binary labels");
}

bool AbstractRuleLearner::canPredictScores(const IRowWiseFeatureMatrix& featureMatrix,
                                           const ITrainingResult& trainingResult) const {
    return this->canPredictScores(featureMatrix, trainingResult.getNumLabels());
}

bool AbstractRuleLearner::canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return this->createRegressionPredictorFactory(featureMatrix, numLabels) != nullptr;
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictScores(featureMatrix, *trainingResult.getRuleModel(), *trainingResult.getLabelSpaceInfo(),
                               trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictScores(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IRegressionPredictorFactory> predictorFactoryPtr =
        this->createRegressionPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        std::unique_ptr<IRegressionPredictor> predictorPtr =
            ruleModel.createRegressionPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictScores(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict regression scores");
}

bool AbstractRuleLearner::canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                                  const ITrainingResult& trainingResult) const {
    return this->canPredictProbabilities(featureMatrix, trainingResult.getNumLabels());
}

bool AbstractRuleLearner::canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
    return this->createProbabilityPredictorFactory(featureMatrix, numLabels) != nullptr;
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const {
    return this->predictProbabilities(featureMatrix, *trainingResult.getRuleModel(),
                                      *trainingResult.getLabelSpaceInfo(), trainingResult.getNumLabels());
}

std::unique_ptr<DensePredictionMatrix<float64>> AbstractRuleLearner::predictProbabilities(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    std::unique_ptr<IProbabilityPredictorFactory> predictorFactoryPtr =
        this->createProbabilityPredictorFactory(featureMatrix, numLabels);

    if (predictorFactoryPtr) {
        std::unique_ptr<IProbabilityPredictor> predictorPtr =
            ruleModel.createProbabilityPredictor(*predictorFactoryPtr, labelSpaceInfo);
        return featureMatrix.predictProbabilities(*predictorPtr, numLabels);
    }

    throw std::runtime_error("The rule learner does not support to predict probability estimates");
}
